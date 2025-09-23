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

        self.blocks: List[Dict[str, torch.Tensor]] = []
        for c in self.clients:
            self.blocks.append({k: v.detach().cpu().clone() for k, v in c.get_block_state().items()})

    def select_clients(self, fraction: float = 1.0) -> List:
        m = max(1, int(round(fraction * len(self.clients))))
        return random.sample(self.clients, m)

    def compute_xbar(self) -> Dict[str, torch.Tensor]:
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
        global_state = {k: v.to(self.device) for k, v in xbar.items()}
        for c in selected_clients:
            x_prev_block = self.blocks[c.cid]
            c.set_block_weights(
                global_state=global_state,
                xbar_prev=xbar,
                x_prev_block=x_prev_block,
            )

    def overwrite_block(self, cid: int, new_state: Dict[str, torch.Tensor]):
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
