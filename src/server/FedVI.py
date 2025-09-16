# src/server/FedVI.py
import random
from typing import Dict, List, Optional

import torch


class FedVIServer:
    def __init__(self, global_model: torch.nn.Module, clients: List, *, device: str):
        self.device = torch.device(device)
        self.global_model = global_model.to(self.device)
        self.clients = clients

    def _global_state(self) -> Dict[str, torch.Tensor]:
        return {k: v.detach().cpu() for k, v in self.global_model.state_dict().items()}

    def select_clients(self, fraction: float = 1.0) -> List:
        m = max(1, int(round(fraction * len(self.clients))))
        return random.sample(self.clients, m)

    def broadcast(self, selected_clients: List) -> Dict[str, torch.Tensor]:
        g_state = self._global_state()
        for c in selected_clients:
            c.set_block_weights(g_state, xbar_block=g_state)
        return g_state

    def _aggregate_weighted(
        self,
        payloads: List[Dict],
        fallback_state: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        new_state: Dict[str, torch.Tensor] = {}
        all_keys = set(fallback_state.keys())

        n_list = [p["num_samples"] for p in payloads]
        total = float(sum(n_list)) if len(n_list) > 0 else 1.0
        base_weights = [n / total for n in n_list]

        for k in all_keys:
            pairs = [(w, p["state"][k]) for w, p in zip(base_weights, payloads) if k in p["state"]]
            if not pairs:
                new_state[k] = fallback_state[k]
                continue

            w_sum = sum(w for w, _ in pairs)
            acc = None
            for w, tensor in pairs:
                w = w / w_sum
                acc = tensor * w if acc is None else acc + tensor * w
            new_state[k] = acc

        return new_state

    def aggregate(self, payloads: List[Dict]) -> None:
        g_state = self._global_state()
        new_state = self._aggregate_weighted(payloads, fallback_state=g_state)
        self.global_model.load_state_dict(new_state, strict=True)

    def run_round(
        self,
        *,
        fraction: float = 1.0,
        local_epochs: int = 1,
        eta_r: float = 0.5,
        lambda_reg: float = 0.0,
        prox_cfg: Optional[dict] = None,
    ) -> Dict:
        selected = self.select_clients(fraction=fraction)
        xbar = self.broadcast(selected)

        payloads = []
        for c in selected:
            out = c.train_one_round(
                local_epochs=local_epochs,
                eta_r=eta_r,
                lambda_reg=lambda_reg,
                prox_cfg=prox_cfg,
            )
            payloads.append(out)

        reg_sum = 0.0
        if lambda_reg > 0.0:
            total_n = sum(p["num_samples"] for p in payloads) or 1
            for p in payloads:
                w = p["num_samples"] / total_n
                for k, t in p["state"].items():
                    d = (t - xbar[k]).float()
                    reg_sum += 0.5 * lambda_reg * w * float((d * d).sum().item())

        self.aggregate(payloads)

        return {
            "selected": [p["cid"] for p in payloads],
            "total_samples": sum(p["num_samples"] for p in payloads),
            "eta_r": eta_r,
            "lambda_reg": lambda_reg,
            "reg_value": reg_sum,
        }

    @torch.no_grad()
    def evaluate_global(self, dataset, *, batch_size: int, device: str, loss_fn) -> Dict[str, float]:
        model = self.global_model
        model.eval()
        dev = torch.device(device)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        total, correct = 0, 0
        total_loss = 0.0
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
