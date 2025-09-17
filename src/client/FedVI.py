import torch
from torch.utils.data import DataLoader
from copy import deepcopy
from typing import Dict, Optional

class FedVIClient:
    def __init__(self, cid, model, dataset, *, lr, batch_size, device, owned_keys):
        self.cid = cid
        self.device = torch.device(device)
        self.model = deepcopy(model).to(self.device)
        self.dataset = dataset
        self.batch_size = batch_size
        self.lr = lr
        self.owned_keys = set(owned_keys)
        for n, p in self.model.named_parameters():
            p.requires_grad = (n in self.owned_keys)

        self._xbar: Optional[Dict[str, torch.Tensor]] = None

    def set_block_weights(self, state_block: Dict[str, torch.Tensor], xbar_block: Dict[str, torch.Tensor] = None):
        to_dev = {k: v.to(self.device) for k, v in state_block.items()}
        self.model.load_state_dict(to_dev, strict=False)
        self._xbar = None if xbar_block is None else {k: v.detach().clone().to(self.device)
                                                      for k, v in xbar_block.items()}

    def get_block_state(self) -> Dict[str, torch.Tensor]:
        sd = self.model.state_dict()
        return {k: v.detach().cpu() for k, v in sd.items() if k in self.owned_keys}

    def _loader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)

    def num_samples(self) -> int:
        return len(self.dataset)
    
    def train_one_round(
        self,
        *,
        local_epochs: int = 1,
        eta_r: float = 0.2,
        lambda_reg: float = 0.1,
    ):
        self.model.train()
        opt = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr)
        loss_fn = torch.nn.CrossEntropyLoss()
        loader = self._loader()

        for _ in range(local_epochs):
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                opt.zero_grad()

                # ∇f̃_i
                logits = self.model(x)
                loss = loss_fn(logits, y)
                loss.backward() # p.grad = ∇f̃_i

                with torch.no_grad():
                    for name, p in self.model.named_parameters():
                        if not p.requires_grad or p.grad is None or name not in self.owned_keys:
                            continue

                        # Fi = ∇f̃_i + λ(x - x̄)
                        Fi = p.grad.clone()
                        if lambda_reg > 0.0 and self._xbar is not None:
                            Fi.add_(lambda_reg * (p.data - self._xbar[name].to(p.device)))

                        # g = η_r * p.grad + Fi
                        g = Fi + eta_r * p.grad
                        p.grad.copy_(g)
                
                opt.step()

        return {
            "cid": self.cid,
            "num_samples": len(self.dataset),
            "state": self.get_block_state(),
        }
