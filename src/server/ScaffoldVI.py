# src/server/ScaffoldVI_Personal.py
import random
from typing import Dict, List, Optional, Callable

import torch


class ScaffoldVIServer:
    """
    Personalized ScaffoldVI Server:
      - 维护每个客户端各自的模型参数 block_i（CPU 上的 state_dict）
      - 维护每个客户端各自的控制变量 c_i（CPU 上的 state_dict）
      - 广播时：给客户端 i 发 (block_i, 0)    # 不使用全局 c，保持“无跨客户端平均”
      - 客户端返回 (Δy_i, Δc_i)：仅写回自己的 block 与 c_i
      - 不对不同客户端的同名参数做任何平均
    """

    def __init__(self, base_model: torch.nn.Module, clients: List, *, device: str, gamma_g: float = 1.0):
        self.device = torch.device(device)
        self.clients = clients
        self.gamma_g = float(gamma_g)

        # 用 base_model 初始化每个客户端的个性化模型与控制变量（都保存在 CPU）
        base_state_cpu = {k: v.detach().cpu() for k, v in base_model.state_dict().items()}
        self.blocks: List[Dict[str, torch.Tensor]] = [
            {k: v.clone() for k, v in base_state_cpu.items()} for _ in range(len(clients))
        ]
        self.ci_list: List[Dict[str, torch.Tensor]] = [
            {k: torch.zeros_like(v, device="cpu") for k, v in base_state_cpu.items()}
            for _ in range(len(clients))
        ]

    # ---------- helpers ---------- #
    def _zeros_like(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: torch.zeros_like(v, device="cpu") for k, v in state.items()}

    def select_clients(self, fraction: float = 1.0) -> List[int]:
        m = len(self.clients)
        s = max(1, int(round(fraction * m)))
        ids = list(range(m))
        return random.sample(ids, s)

    def compute_xbar(self) -> Dict[str, torch.Tensor]:
        """\bar{x} = 所有个性化 block 的简单平均（CPU 张量），不会回写。"""
        keys = self.blocks[0].keys()
        xbar = {
            k: torch.stack([b[k] for b in self.blocks], dim=0).mean(dim=0).clone()
            for k in keys
        }
        return xbar  # CPU tensors


    def broadcast(self, selected_ids: List[int], xbar: Dict[str, torch.Tensor] | None = None) -> None:
        """
        给每个被选中的 client i 下发它自己的模型 block_i；
        为了不引入“全局 c”，这里给 c_global 传全零（同形）。
        """
        for i in selected_ids:
            g_state = self.blocks[i]                                   # 个性化模型
            zero_c = self._zeros_like(g_state)                         # 不使用全局 c
            self.clients[i].set_broadcast(g_state, zero_c, xbar_state=xbar)

    # ---------- one federated round ---------- #
    def run_round(
        self,
        *,
        fraction: float = 1.0,
        local_epochs: int = 1,
        eta_r: float = 0.5,
        Fi: Optional[Callable[..., Dict[str, torch.Tensor]]] = None,
        fi_kwargs: Optional[dict] = None,
        clip_grad: Optional[float] = None,
        lambda_reg: float = 0.1,
        round_idx: int = 1,
    ) -> Dict:
        """
        个性化训练：仅写回被选中客户端自己的 block 和 c_i。
        """
        sel_ids = self.select_clients(fraction=fraction)

        xbar = self.compute_xbar()
        self.broadcast(sel_ids, xbar=xbar)

        payloads = []
        for i in sel_ids:
            out = self.clients[i].train_one_round(
                local_epochs=local_epochs,
                eta_r=eta_r,
                Fi=Fi,
                fi_kwargs=fi_kwargs,
                clip_grad=clip_grad,
                lambda_reg=lambda_reg,
            )
            payloads.append((i, out))

        # 覆盖写回（无跨客户端平均）
        for i, out in payloads:
            dy: Dict[str, torch.Tensor] = out["delta_y"]  # CPU tensors
            dc: Dict[str, torch.Tensor] = out["delta_c"]
            # x_i ← x_i + γ_g · Δy_i
            new_block = {}
            for k, v in self.blocks[i].items():
                new_block[k] = v + self.gamma_g * dy[k]
            self.blocks[i] = new_block
            # c_i ← c_i + Δc_i  （服务器也留一份，便于可视化/保存）
            new_ci = {}
            for k, v in self.ci_list[i].items():
                new_ci[k] = v + dc[k]
            self.ci_list[i] = new_ci

        return {
            "selected": [i for i, _ in payloads],
            "total_samples": sum(out["num_samples"] for _, out in payloads),
            "s_size": len(sel_ids),
            "m_clients": len(self.clients),
            "eta_r": eta_r,
            "gamma_g": self.gamma_g,
            "lambda_reg": lambda_reg,
        }

    def global_reg_value(self, lambda_reg: float) -> float:
        """监控 ∑ λ/2 ||x_i - \bar{x}||^2（仅用于打印）。"""
        xbar = self.compute_xbar()
        val = 0.0
        for b in self.blocks:
            for k in b.keys():
                d = (b[k] - xbar[k]).float()
                val += 0.5 * lambda_reg * float((d * d).sum().item())
        return val

    # ---------- inspection / utilities ---------- #
    def get_block(self, client_id: int) -> Dict[str, torch.Tensor]:
        """拿到客户端 i 的个性化模型参数（CPU 张量）"""
        return self.blocks[client_id]

    def get_ci(self, client_id: int) -> Dict[str, torch.Tensor]:
        """拿到客户端 i 的控制变量（CPU 张量）"""
        return self.ci_list[client_id]

    @torch.no_grad()
    def evaluate_personalized(self, test_subsets: List, *, batch_size: int, device: str, loss_fn) -> Dict:
        """
        个性化评估：每个客户端用“自己的模型参数”在“自己的测试子集”上评估；
        返回加权总体与 per-client 明细。
        """
        total_loss, total_correct, total_n = 0.0, 0, 0
        per_client = []
        for i, (cli, testset) in enumerate(zip(self.clients, test_subsets)):
            # 把个性化 block_i 加载到该 client 的模型上评估
            cli.set_broadcast(self.blocks[i], self._zeros_like(self.blocks[i]))
            res = cli.evaluate(testset, batch_size=batch_size, device=device, loss_fn=loss_fn)
            n = res["num_samples"]
            total_loss += res["loss"] * n
            total_correct += int(res["accuracy"] * n)
            total_n += n
            per_client.append({"cid": i, **res})

        overall = {
            "loss": total_loss / total_n if total_n > 0 else 0.0,
            "accuracy": total_correct / total_n if total_n > 0 else 0.0,
            "num_samples": total_n,
        }
        return {"overall": overall, "per_client": per_client}

    # （可选）没有“全局模型”概念，若有人误用 evaluate_global，就明确提示
    def evaluate_global(self, *args, **kwargs):
        raise NotImplementedError(
            "Personalized server has no single global model. "
            "Use `evaluate_personalized(...)` instead."
        )
