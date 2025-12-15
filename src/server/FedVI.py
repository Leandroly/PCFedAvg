# src/server/FedVI.py
import torch
import random
import math
import numpy as np
from typing import Dict, List
from src.utils.logger import TraceLogger, vectorize_owned


class FedVIServer:
    def __init__(self, global_model: torch.nn.Module, clients: List, *, device: str, overwrite=False):
        self.device = torch.device(device)
        self.global_model = global_model.to(self.device)
        self.clients = clients
        self._round = 0
        self.logger = TraceLogger(log_file="fedvi_trace.log", overwrite=overwrite)

        # 每个 client 自己的 block 参数（不包含未拥有的层）
        self.blocks: List[Dict[str, torch.Tensor]] = []
        for c in self.clients:
            self.blocks.append({k: v.detach().cpu().clone() for k, v in c.get_block_state().items()})

    # --------- 基本工具函数 ---------

    def select_clients(self, fraction: float = 1.0) -> List:
        m = max(1, int(round(fraction * len(self.clients))))
        return random.sample(self.clients, m)

    def compute_xbar(self) -> Dict[str, torch.Tensor]:
        """计算当前各 client block 的简单平均 \bar{x}"""
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
        """把上一轮的 \bar{x}^{r-1} 和各自的 x_i^{r-1} 发给被选中的 client"""
        for c in selected_clients:
            x_prev_block = self.blocks[c.cid]
            others = [self.blocks[j] for j in range(len(self.blocks)) if j != c.cid]
            c.set_block_weights(
                xbar_prev=xbar,
                x_prev_block=x_prev_block,
                others_prev_blocks=others,
            )

    def _broadcast_global_snapshot(self):
        """让每个 client 知道其它 client 的 block（只作为缓存，不覆盖自己参数）"""
        for c in self.clients:
            c.update_from_global(self.blocks)

    def overwrite_block(self, cid: int, new_state: Dict[str, torch.Tensor]):
        """server 只更新该 client 的 block 记录"""
        self.blocks[cid] = {k: v.detach().cpu().clone() for k, v in new_state.items()}

    # --------- 记录初始化信息 ---------

    def log_round0(self, max_show: int = 5):
        """把 Round 0 的初始化参数写到 log 文件和终端"""
        all_cids = [c.cid for c in self.clients]

        baselines = {}
        vec_dim0 = 0
        for c in self.clients:
            cid = c.cid
            state = self.blocks[cid]   # 初始 state_dict（只含 owned keys）
            vec = vectorize_owned(state, c.owned_keys)
            if vec_dim0 == 0:
                vec_dim0 = int(vec.numel())
            head = [float(x) for x in vec[:max_show].tolist()] if vec.numel() > 0 else []
            baselines[cid] = {
                "norm": float(vec.norm().item()) if vec.numel() > 0 else 0.0,
                "head": head,
            }

        # Round 0 没有选中 client，payloads=[]
        self.logger.log_round(
            round_id=0,
            payloads=[],
            all_client_ids=all_cids,
            vec_dim=vec_dim0,
            baselines=baselines,
            step_count=0,
        )

    # --------- 计算当前的 drift（用于 rho 校准） ---------

    @torch.no_grad()
    def _compute_current_drifts(self) -> List[float]:
        """
        计算当前所有 client 的 drift:
            d_i = || x_i - xbar ||_2
        这里的 x_i 就是 self.blocks[cid]，xbar 是 compute_xbar()
        """
        xbar = self.compute_xbar()
        drifts: List[float] = []
        for cid, block in enumerate(self.blocks):
            diff_sq = 0.0
            for k, v in block.items():
                # v: x_i^k, xbar[k]: \bar{x}^k
                diff = v - xbar[k]
                diff_sq += (diff ** 2).sum().item()
            drifts.append(float(diff_sq ** 0.5))
        return drifts

    def calibrate_rho(
        self,
        warmup_rounds: int = 2,
        *,
        fraction: float = 1.0,
        local_steps: int = 1,
        eta_r: float = 0.2,
    ) -> Dict[str, float]:
        """
        工具函数：先跑若干轮“基本无约束”的 warmup，
        统计 drift d_i = ||x_i - xbar|| 的典型尺度，
        返回三档 rho: tight / mid / loose。

        用法示例：
            stats = server.calibrate_rho(warmup_rounds=2)
            rho_tight = stats["rho_tight"]
            rho_mid   = stats["rho_mid"]
            rho_loose = stats["rho_loose"]
        """
        all_drifts: List[float] = []

        for _ in range(warmup_rounds):
            # 设置一个非常大的 rho，基本不会触发 penalty
            self.run_round(
                fraction=fraction,
                local_steps=local_steps,
                eta_r=eta_r,
                rho=1e9,
            )
            round_drifts = self._compute_current_drifts()
            all_drifts.extend(round_drifts)

        if not all_drifts:
            # 没有 client 或没有 drift，退化回 1.0
            median_drift = 1.0
        else:
            median_drift = float(np.median(all_drifts))

        # 三档 rho：可以根据需要修改倍率
        rho_mid = median_drift
        rho_tight = 0.5 * median_drift
        rho_loose = 1.5 * median_drift

        return {
            "median_drift": median_drift,
            "rho_tight": rho_tight,
            "rho_mid": rho_mid,
            "rho_loose": rho_loose,
        }
    
    @torch.no_grad()
    def compute_drifts(self) -> Dict[int, float]:
        """
        返回 {cid: drift = ||x_i - xbar|| } for all clients
        """
        xbar = self.compute_xbar()
        result = {}
        for c in self.clients:
            cid = c.cid
            block = self.blocks[cid]
            sq = 0.0
            for k in block.keys():
                diff = block[k].to(self.device) - xbar[k].to(self.device)
                sq += (diff * diff).sum().item()
            result[cid] = math.sqrt(sq)
        return result

    # --------- 一轮训练 ---------

    def run_round(
        self,
        *,
        fraction: float = 1.0,
        local_steps: int = 1,
        eta_r: float = 0.2,
        rho: float = 1.0,
    ) -> Dict:
        
        self._round += 1
        selected = self.select_clients(fraction=fraction)

        # 记录当前各 client 的 baseline norm（用于 trace）
        baselines = {}
        for c in self.clients:
            cid = c.cid
            state = self.blocks[cid]
            vec = vectorize_owned(state, c.owned_keys)

            head = [float(x) for x in vec[:5].tolist()] if vec.numel() > 0 else []
            baselines[cid] = {
                "norm": float(vec.norm().item()) if vec.numel() > 0 else 0.0,
                "head": head,
            }

        # 这一轮开始前的块平均 \bar{x}^{r-1}
        xbar_prev = self.compute_xbar()
        self._broadcast_round_state(selected, xbar_prev)

        payloads = []
        for c in selected:
            out = c.train_one_round(
                local_steps=local_steps,
                eta_r=eta_r,
                rho=rho,
            )
            payloads.append(out)
            self.overwrite_block(c.cid, out["state"])

        all_cids = [c.cid for c in self.clients]
        vec_dim = payloads[0].get("vec_dim", 0) if payloads else 0

        self.logger.log_round(
            self._round,
            payloads,
            all_client_ids=all_cids,
            vec_dim=vec_dim,
            baselines=baselines,
            step_count=local_steps,
            eta_r=eta_r,
            rho=rho,
        )

        # 只更新各 client 对“其它 client block”的快照，不覆盖本地参数
        self._broadcast_global_snapshot()

        return {
            "selected": [p["cid"] for p in payloads],
            "total_samples": sum(p["num_samples"] for p in payloads),
        }

    # --------- 评估函数 ---------

    @torch.no_grad()
    def evaluate_global(self, dataset, *, batch_size: int, device: str, loss_fn) -> Dict[str, float]:
        """
        用当前 block 平均 \bar{x} 作为 global model 的参数，在给定 dataset 上评估。
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

    @torch.no_grad()
    def evaluate_locals(self, *, batch_size: int, device: str, loss_fn):
        """
        用每个 client 自己的 block 参数，在自己的本地数据上评估。
        """
        dev = torch.device(device)
        model = self.global_model
        results = []

        for c in self.clients:
            # 加载该 client 自己的参数
            state_c = {k: v.to(self.device) for k, v in self.blocks[c.cid].items()}
            model.load_state_dict(state_c, strict=True)
            model.eval()

            # 用该 client 的本地数据评估
            loader = torch.utils.data.DataLoader(
                c.dataset, batch_size=batch_size, shuffle=False, num_workers=0
            )
            total, correct, total_loss = 0, 0, 0.0
            for x, y in loader:
                x, y = x.to(dev), y.to(dev)
                logits = model(x)
                loss = loss_fn(logits, y)
                total_loss += loss.item() * y.size(0)
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)

            results.append({
                "cid": c.cid,
                "loss": (total_loss / total) if total > 0 else 0.0,
                "accuracy": (correct / total) if total > 0 else 0.0,
                "num_samples": total,
            })

        return results
