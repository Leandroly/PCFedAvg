# src/server/FedVI.py
import random
from typing import Dict, List

import torch


class FedVIServer:
    def __init__(self, global_model: torch.nn.Module, clients: List, *, device: str):
        self.device = torch.device(device)
        self.global_model = global_model.to(self.device)
        self.clients = clients

        self.blocks: List[Dict[str, torch.Tensor]] = []
        for c in self.clients:
            self.blocks.append({k: v.detach().cpu().clone() for k, v in c.get_block_state().items()})

    def select_clients(self, fraction: float = 1.0) -> List:
        m = max(1, int(round(fraction * len(self.clients))))
        return random.sample(self.clients, m)

    def compute_xbar(self) -> Dict[str, torch.Tensor]:
        keys = self.blocks[0].keys()
        xbar: Dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for k in keys:
                acc = None
                for b in self.blocks:
                    t = b[k]
                    acc = t.clone() if acc is None else acc.add(t)
                xbar[k] = acc.div(float(len(self.blocks)))
        return xbar

    def send_xbar_only(self, selected_clients: List, xbar: Dict[str, torch.Tensor]):
        empty = {}
        for c in selected_clients:
            c.set_block_weights(empty, xbar_block=xbar)

    def overwrite_block(self, cid: int, new_state: Dict[str, torch.Tensor]):
        self.blocks[cid] = {k: v.detach().cpu().clone() for k, v in new_state.items()}

    def run_round(
        self,
        *,
        fraction: float = 1.0,
        local_epochs: int = 1,
        eta_r: float = 0.2,
        lambda_reg: float = 0.1,
    ) -> Dict:
        selected = self.select_clients(fraction=fraction)

        xbar = self.compute_xbar()
        self.send_xbar_only(selected, xbar)

        payloads = []
        for c in selected:
            out = c.train_one_round(
                local_epochs=local_epochs,
                eta_r=eta_r,
                lambda_reg=lambda_reg,
            )
            payloads.append(out)
            self.overwrite_block(c.cid, out["state"])

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
