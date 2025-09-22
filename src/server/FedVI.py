# src/server/FedVI.py
import torch
import random
from typing import Dict, List
from src.utils.logger import TraceLogger, vectorize_owned



class FedVIServer:
    def __init__(self, global_model: torch.nn.Module, clients: List, *, device: str, overwrite=False):
        self.device = torch.device(device)
        self.global_model = global_model.to(self.device)
        self.clients = clients
        self._round = 0
        self.logger = TraceLogger(log_file="fedvi_trace.log", overwrite=overwrite)

        # 按客户端缓存“上一轮各自块”的参数（owned_keys 子集）
        self.blocks: List[Dict[str, torch.Tensor]] = []
        for c in self.clients:
            self.blocks.append({k: v.detach().cpu().clone() for k, v in c.get_block_state().items()})

    def select_clients(self, fraction: float = 1.0) -> List:
        m = max(1, int(round(fraction * len(self.clients))))
        return random.sample(self.clients, m)

    def compute_xbar(self) -> Dict[str, torch.Tensor]:
        """
        均匀平均（\bar x = (1/m) * sum_j x_j），
        对 self.blocks 里的每个参数名逐元素求均值，得到“均值模型” state_dict。
        """
        assert len(self.blocks) > 0, "No client blocks cached."
        keys = self.blocks[0].keys()
        m = float(len(self.blocks))
        xbar: Dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for k in keys:
                acc = None
                for b in self.blocks:
                    t = b[k]
                    acc = t.clone() if acc is None else (acc + t)
                xbar[k] = acc / m
        return xbar

    def _broadcast_round_state(self, selected_clients: List, xbar: Dict[str, torch.Tensor]):
        """
        向本轮选中的每个 client 下发：
          1) global_state := x^{r-1} 的“均值模型”（全量 dict）
          2) xbar_prev    := 同样的均值模型（全量 dict）
          3) x_prev_block := 该 client 自己上一轮的块（owned_keys 子集）
        client 侧会据此计算动态 \bar x_{i,k-1}^r，并且只更新 owned_keys。
        """
        global_state = {k: v.to(self.device) for k, v in xbar.items()}
        for c in selected_clients:
            x_prev_block = self.blocks[c.cid]  # 仅 owned_keys
            c.set_block_weights(
                global_state=global_state,
                xbar_prev=xbar,               # 全量；client 只会用到 owned_keys
                x_prev_block=x_prev_block,
            )

    def overwrite_block(self, cid: int, new_state: Dict[str, torch.Tensor]):
        """
        用 client cid 上传的 owned_keys 覆盖其缓存块。
        """
        self.blocks[cid] = {k: v.detach().cpu().clone() for k, v in new_state.items()}

    def run_round(
        self,
        *,
        fraction: float = 1.0,
        local_steps: int = 1,
        eta_r: float = 0.2,
        lambda_reg: float = 0.1,
    ) -> Dict:
        
        self._round += 1
        selected = self.select_clients(fraction=fraction)

        # === baselines：本轮开始前各 client 的块向量（用于未选中的静止轨迹）===
        baselines = {}
        for c in self.clients:
            cid = c.cid
            state = self.blocks[cid]  # x^{r-1,i} 的 state_dict（CPU 张量）

            # 找第一组 2D 权重和匹配的 1D bias
            weight_key, bias_key, out_features = None, None, None
            for k, t in state.items():
                if t.ndim == 2:
                    weight_key = k
                    out_features = t.shape[0]
                    break
            if weight_key is not None:
                for k, t in state.items():
                    if t.ndim == 1 and t.shape[0] == out_features:
                        bias_key = k
                        break

            # 取该 client 的行（与 client 相同规则）
            if weight_key is not None and bias_key is not None:
                row = int(c.cid % out_features)
                w_row = state[weight_key][row, :].view(-1)
                b_one = state[bias_key][row].view(-1)
                vec = torch.cat([w_row, b_one])  # 785
            else:
                vec = vectorize_owned(state, c.owned_keys)  # 兜底

            head = [float(x) for x in vec[:5].tolist()] if vec.numel() > 0 else []
            baselines[cid] = {
                "norm": float(vec.norm().item()) if vec.numel() > 0 else 0.0,
                "head": head,
            }

        xbar_prev = self.compute_xbar()
        self._broadcast_round_state(selected, xbar_prev)

        payloads = []
        for c in selected:
            out = c.train_one_round(
                local_steps=local_steps,
                eta_r=eta_r,
                lambda_reg=lambda_reg,
            )
            payloads.append(out)
            self.overwrite_block(c.cid, out["state"])

        all_cids = [c.cid for c in self.clients]
        vec_dim = payloads[0].get("vec_dim", 0) if payloads else 0

        # 终端摘要 + 文件矩阵（包含所有 clients）
        self.logger.log_round(
            self._round,
            payloads,
            all_client_ids=all_cids,
            vec_dim=vec_dim,
            baselines=baselines,
            step_count=local_steps,
        )

        return {
            "selected": [p["cid"] for p in payloads],
            "total_samples": sum(p["num_samples"] for p in payloads),
        }

    @torch.no_grad()
    def evaluate_global(self, dataset, *, batch_size: int, device: str, loss_fn) -> Dict[str, float]:
        """
        用“均值模型”（\bar x）评估（与你原来的逻辑保持一致）。
        如需评估其它汇总方式，可在此更换 compute_xbar 的定义。
        """
        xbar = self.compute_xbar()
        to_dev = {k: v.to(self.device) for k, v in xbar.items()}
        self.global_model.load_state_dict(to_dev, strict=True)

        model = self.global_model
        model.eval()
        dev = torch.device(device)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        total, correct, total_loss = 0, 0, 0.0
        for x, y in loader:
            x, y = x.to(dev), y.to(dev)
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
