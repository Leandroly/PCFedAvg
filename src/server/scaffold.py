import random
import torch
from typing import Dict, List

class ScaffoldServer:

    def __init__(self, global_model: torch.nn.Module, clients: List, *, device: str = "cuda"):
        self.device = torch.device(device)
        self.global_model = global_model.to(self.device)
        self.clients = clients

        with torch.no_grad():
            self.c_global: Dict[str, torch.Tensor] = {
                k: torch.zeros_like(v, device=self.device)
                for k, v in self.global_model.state_dict().items()
            }

    def _global_state(self) -> Dict[str, torch.Tensor]:
        return {k: v.detach().cpu() for k, v in self.global_model.state_dict().items()}

    def _global_control(self) -> Dict[str, torch.Tensor]:
        return {k: v.detach().cpu() for k, v in self.c_global.items()}

    def select_clients(self, fraction: float = 1.0) -> List:
        m = max(1, int(fraction * len(self.clients)))
        return random.sample(self.clients, m)

    def broadcast(self, selected_clients: List) -> None:
        g_state = self._global_state()
        c_state = self._global_control()
        for client in selected_clients:
            if hasattr(client, "set_global_weights"):
                client.set_global_weights(g_state)
            else:
                client.model.load_state_dict({k: v.to(client.device) for k, v in g_state.items()}, strict=True)

            if hasattr(client, "set_global_control"):
                client.set_global_control(c_state)
            else:
                with torch.no_grad():
                    client.c_global = {k: v.to(client.device) for k, v in c_state.items()}

    def _aggregate_state(self, payloads: List[Dict]) -> Dict[str, torch.Tensor]:
        total = sum(p["num_samples"] for p in payloads)
        keys = payloads[0]["state"].keys()
        new_state = {}
        for k in keys:
            acc = None
            for p in payloads:
                w = p["num_samples"] / total
                t = p["state"][k]
                acc = t * w if acc is None else acc + t * w
            new_state[k] = acc
        return new_state

    def _aggregate_delta_c(self, payloads: List[Dict]) -> Dict[str, torch.Tensor]:
        total = sum(p["num_samples"] for p in payloads)
        keys = None
        for p in payloads:
            if "delta_c" in p and p["delta_c"] is not None:
                keys = p["delta_c"].keys()
                break
        if keys is None:
            return {}

        agg = {}
        for k in keys:
            acc = None
            for p in payloads:
                if "delta_c" not in p or p["delta_c"] is None:
                    continue
                w = p["num_samples"] / total
                t = p["delta_c"][k]
                acc = t * w if acc is None else acc + t * w
            if acc is not None:
                agg[k] = acc
        return agg

    def aggregate(self, payloads: List[Dict]) -> None:
        new_state = self._aggregate_state(payloads)
        self.global_model.load_state_dict({k: v.to(self.device) for k, v in new_state.items()}, strict=True)

        delta_c_agg = self._aggregate_delta_c(payloads)
        if delta_c_agg:
            with torch.no_grad():
                for k, v in delta_c_agg.items():
                    self.c_global[k].add_(v.to(self.device))

    def run_round(self, *, fraction: float = 1.0, local_epochs: int = 1) -> Dict:
        selected = self.select_clients(fraction)
        self.broadcast(selected)
        payloads = [c.train_one_round(local_epochs=local_epochs) for c in selected]
        self.aggregate(payloads)
        return {
            "selected": [p["cid"] for p in payloads],
            "total_samples": sum(p["num_samples"] for p in payloads),
        }

    @torch.no_grad()
    def evaluate_global(self, dataset, *, batch_size: int, loss_fn, device: str = None) -> Dict[str, float]:
        dev = torch.device(device) if device is not None else self.device
        model = self.global_model
        model.eval()
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        total = 0
        correct = 0
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
