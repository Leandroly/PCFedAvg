# src/server/ScaffoldVI_Bilevel.py
import random
from typing import Dict, List

import torch


class ScaffoldVIServer:
    def __init__(self, base_model: torch.nn.Module, clients: List, *, device: str, gamma_g: float = 0.2):
        self.device = torch.device(device)
        self.clients = clients
        self.gamma_g = float(gamma_g)

        # base state on CPU
        base_state = {k: v.detach().cpu() for k, v in base_model.state_dict().items()}

        # global variables (CPU tensors)
        self.global_x: Dict[str, torch.Tensor] = {k: v.clone() for k, v in base_state.items()}
        self.global_c: Dict[str, torch.Tensor] = {k: torch.zeros_like(v) for k, v in base_state.items()}

        # personalized blocks (CPU tensors)
        n = len(clients)
        self.blocks: List[Dict[str, torch.Tensor]] = [
            {k: v.clone() for k, v in base_state.items()} for _ in range(n)
        ]
        self.ci_list: List[Dict[str, torch.Tensor]] = [
            {k: torch.zeros_like(v) for k, v in base_state.items()} for _ in range(n)
        ]

    # ---------------- helpers ---------------- #
    def _zeros_like(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: torch.zeros_like(v) for k, v in state.items()}

    def select_clients(self, fraction: float = 1.0) -> List[int]:
        m = len(self.clients)
        s = max(1, int(round(fraction * m)))
        return random.sample(list(range(m)), s)

    def _avg_dict(self, dicts: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """elementwise average over a list of CPU state-dicts (assumed same keys)"""
        assert len(dicts) > 0
        keys = dicts[0].keys()
        out = {}
        for k in keys:
            acc = None
            for d in dicts:
                acc = d[k].clone() if acc is None else acc.add(d[k])
            out[k] = acc.div(float(len(dicts)))
        return out

    # ---------------- communication ---------------- #
    def broadcast(self, selected_ids: List[int]) -> None:
        for i in selected_ids:
            self.clients[i].set_broadcast(
                global_state=self.blocks[i],          # y_{i,0} ← x_i
                global_c=self.global_c,               # c_g
                xbar_state=self.global_x,             # 用于 λ‖x_i - x^g‖^2
            )

    # ---------------- one round ---------------- #
    def run_round(
        self,
        *,
        fraction: float = 1.0,
        local_epochs: int = 1,
        eta_r: float = 0.2,
        lambda_reg: float = 0.1,
    ) -> Dict:
        sel_ids = self.select_clients(fraction=fraction)

        # 下发各自个性化起点 + 全局 c_g + 全局 x_g(作正则中心)
        self.broadcast(sel_ids)

        # 本地训练并收集增量
        payloads = []
        for i in sel_ids:
            out = self.clients[i].train_one_round(
                local_epochs=local_epochs,
                eta_r=eta_r,
                lambda_reg=lambda_reg,
            )
            payloads.append((i, out))

        # 覆盖写回个性化 block：x_i ← x_i + γ_g Δy_i,  c_i ← c_i + Δc_i
        for i, out in payloads:
            dy: Dict[str, torch.Tensor] = out["delta_y"]  # CPU
            dc: Dict[str, torch.Tensor] = out["delta_c"]  # CPU

            new_xi = {k: self.blocks[i][k] + self.gamma_g * dy[k] for k in self.blocks[i].keys()}
            new_ci = {k: self.ci_list[i][k] + dc[k] for k in self.ci_list[i].keys()}

            self.blocks[i]  = new_xi
            self.ci_list[i] = new_ci

        # 计算全局平均增量（原 SCAFFOLD 公式：简单平均）
        delta_y_list = [out["delta_y"] for _, out in payloads]
        delta_c_list = [out["delta_c"] for _, out in payloads]
        if len(delta_y_list) > 0:
            Dx = self._avg_dict(delta_y_list)   # Δx^r
            Dc = self._avg_dict(delta_c_list)   # Δc^r
        else:
            Dx = {k: torch.zeros_like(v) for k, v in self.global_x.items()}
            Dc = {k: torch.zeros_like(v) for k, v in self.global_c.items()}

        # 全局更新： x^g ← x^g + γ_g Δx^r
        for k in self.global_x.keys():
            self.global_x[k].add_(self.gamma_g * Dx[k])

        # 全局控制变量更新： c^g ← c^g + (|S|/m) Δc^r
        factor = float(len(sel_ids)) / float(len(self.clients))
        for k in self.global_c.keys():
            self.global_c[k].add_(factor * Dc[k])

        return {
            "selected": sel_ids,
            "total_samples": int(sum(out["num_samples"] for _, out in payloads)),
            "s_size": len(sel_ids),
            "m_clients": len(self.clients),
            "eta_r": eta_r,
            "gamma_g": self.gamma_g,
            "gamma_l": (self.clients[sel_ids[0]].lr if sel_ids else 0.0),
            "lambda_reg": lambda_reg,
        }

    # ---------------- inspection / evaluation ---------------- #
    def get_block(self, client_id: int) -> Dict[str, torch.Tensor]:
        return self.blocks[client_id]

    def get_ci(self, client_id: int) -> Dict[str, torch.Tensor]:
        return self.ci_list[client_id]

    def get_global_state(self) -> Dict[str, torch.Tensor]:
        return self.global_x

    @torch.no_grad()
    def evaluate_global(self, dataset, *, batch_size: int, loss_fn) -> Dict[str, float]:
        """评估全局模型 x^g（不是个性化）"""
        # 用一份模型承载 global_x 评估
        # 注意：把张量搬到 device 上再 load
        dummy = next(iter(self.clients)).model if hasattr(self.clients[0], "model") else None
        if dummy is None:
            raise RuntimeError("Server needs a reference model from clients for evaluation.")
        model = deepcopy_on_device(dummy, self.device)
        model.load_state_dict({k: v.to(self.device) for k, v in self.global_x.items()}, strict=True)
        model.eval()

        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        total, correct, total_loss = 0, 0, 0.0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            logits = model(x)
            loss = loss_fn(logits, y)
            total_loss += loss.item() * y.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

        return {
            "loss": (total_loss / total) if total > 0 else 0.0,
            "accuracy": (correct / total) if total > 0 else 0.0,
            "num_samples": total,
        }


def deepcopy_on_device(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    import copy
    m = copy.deepcopy(model).to(device)
    m.eval()
    return m
