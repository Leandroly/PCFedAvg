import torch
from torch.utils.data import DataLoader
from copy import deepcopy
from typing import Dict, List
from src.utils.logger import vectorize_owned, make_trace_entry
from itertools import cycle


class FedVIClient:
    def __init__(self, cid, model, dataset, *, lr, batch_size, device, owned_keys, m_total: int):
        self.cid = cid
        self.device = torch.device(device)
        self.model = deepcopy(model).to(self.device)
        self.dataset = dataset
        self.batch_size = batch_size
        self.lr = lr
        self.owned_keys = set(owned_keys)
        self.m_total = int(m_total)

        for n, p in self.model.named_parameters():
            p.requires_grad = (n in self.owned_keys)

        self._xbar_prev: Dict[str, torch.Tensor] = {}      # \bar{x}^{r-1}
        self._x_prev_block: Dict[str, torch.Tensor] = {}   # x^{r-1,i}
        self._others_prev_blocks: List[Dict[str, torch.Tensor]] | None = None

    def set_block_weights(
        self,
        #global_state: Dict[str, torch.Tensor],
        xbar_prev: Dict[str, torch.Tensor],
        x_prev_block: Dict[str, torch.Tensor],
        others_prev_blocks: List[Dict[str, torch.Tensor]] | None = None,
    ):
        #to_dev = {k: v.to(self.device) for k, v in global_state.items()}
        #self.model.load_state_dict(to_dev, strict=True)

        self._xbar_prev = {k: v.detach().clone().to(self.device) for k, v in xbar_prev.items()}
        self._x_prev_block = {k: v.detach().clone().to(self.device) for k, v in x_prev_block.items()}
        if others_prev_blocks is not None:
            self._others_prev_blocks = [
                {k: v.detach().clone().to(self.device) for k, v in row.items()}
                for row in others_prev_blocks
            ]

    def update_from_global(self, global_blocks: List[Dict[str, torch.Tensor]]):
        cache = []
        for j, row in enumerate(global_blocks):
            if j == self.cid:
                cache.append(self._x_prev_block)
            else:
                cache.append({k: v.detach().clone().to(self.device) for k, v in row.items()})
        self._others_prev_blocks = cache

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
        local_steps: int = 1,
        eta_r: float = 0.2,
        lambda_reg: float = 0.1,
    ):
        trace = []
        vec_dim = 0
        self.model.train()
        opt = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr)
        loss_fn = torch.nn.CrossEntropyLoss()
        loader = self._loader()

        m = self.m_total
        it = cycle(loader)

        # k=0
        sd0 = self.model.state_dict()
        y_vec0 = vectorize_owned(sd0, self.owned_keys)
        vec_dim = int(y_vec0.numel())
        rec0 = make_trace_entry(0, y_vec0, max_show=5)
        if rec0:
            trace.append(rec0)

        for step in range(local_steps):
            x, y = next(it)
            x, y = x.to(self.device), y.to(self.device)
            opt.zero_grad()

            # forward & loss
            logits = self.model(x)
            loss = loss_fn(logits, y)
            loss.backward()  # p.grad = ∇f̃_i

            with torch.no_grad():
                for name, p in self.model.named_parameters():
                    if not (p.requires_grad and name in self.owned_keys and p.grad is not None):
                        continue

                    grad_ftilde = p.grad  # ∇f̃_i

                    # 动态 \bar{x}_{i,k-1}^r
                    xbar_local = self._xbar_prev[name] + (p.data - self._x_prev_block[name]) / float(m)

                    # prox 系数 λ*(m-1)/m
                    prox_coeff = lambda_reg * (float(m) - 1.0) / float(m)

                    # g = (1+η_r)*∇f̃_i + λ*(m-1)/m * (x_i - x̄_local)
                    g = (1.0 + eta_r) * grad_ftilde + prox_coeff * (p.data - xbar_local)

                    p.grad.copy_(g)

            opt.step()

            sd = self.model.state_dict()
            y_vec = vectorize_owned(sd, self.owned_keys)
            vec_dim = int(y_vec.numel())

            rec = make_trace_entry(step + 1, y_vec, max_show=5)
            if rec:
                trace.append(rec)

        return {
            "cid": self.cid,
            "num_samples": len(self.dataset),
            "state": self.get_block_state(),
            "trace": trace,
            "vec_dim": vec_dim,
        }