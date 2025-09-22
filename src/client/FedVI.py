import torch
from torch.utils.data import DataLoader
from copy import deepcopy
from typing import Dict
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
        self.m_total = int(m_total)   # 总 client 数 m

        for n, p in self.model.named_parameters():
            p.requires_grad = (n in self.owned_keys)

        # 存储本轮需要的全局信息
        self._xbar_prev: Dict[str, torch.Tensor] = {}      # \bar{x}^{r-1}
        self._x_prev_block: Dict[str, torch.Tensor] = {}   # x^{r-1,i}

    def set_block_weights(
        self,
        global_state: Dict[str, torch.Tensor],
        xbar_prev: Dict[str, torch.Tensor],
        x_prev_block: Dict[str, torch.Tensor],
    ):
        """
        每一轮通信时调用：
          global_state: 全局模型 x^{r-1}
          xbar_prev:    \bar{x}^{r-1} （所有参数）
          x_prev_block: client i 的上一轮块参数 x^{r-1,i}
        """
        to_dev = {k: v.to(self.device) for k, v in global_state.items()}
        self.model.load_state_dict(to_dev, strict=True)

        self._xbar_prev = {k: v.detach().clone().to(self.device) for k, v in xbar_prev.items()}
        self._x_prev_block = {k: v.detach().clone().to(self.device) for k, v in x_prev_block.items()}

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
        it = cycle(loader)  # 无限迭代 dataloader，保证能取到 local_steps 个 batch

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

            # ===== 记录更新后的 y：只取自己一行 + 对应 bias（784 + 1 = 785）=====
            sd = self.model.state_dict()

            # 找第一组 2D 权重和匹配的 1D bias（比如 Linear 的 weight[10,784] / bias[10]）
            weight_key, bias_key, out_features = None, None, None
            for k, t in sd.items():
                if t.ndim == 2:              # 例如 [num_classes, in_dim]
                    weight_key = k
                    out_features = t.shape[0]
                    break
            if weight_key is not None:
                for k, t in sd.items():
                    if t.ndim == 1 and t.shape[0] == out_features:
                        bias_key = k
                        break

            # 取本 client 的行（row = cid % out_features）
            if weight_key is not None and bias_key is not None:
                row = int(self.cid % out_features)
                w_row = sd[weight_key][row, :].detach().view(-1).cpu()
                b_one = sd[bias_key][row].detach().view(-1).cpu()
                y_vec = torch.cat([w_row, b_one])          # 785 维
            else:
                # 兜底：如果没找到，就退回到旧逻辑（整层向量化）
                y_vec = vectorize_owned(sd, self.owned_keys)

            vec_dim = int(y_vec.numel())                   # 现在应为 785
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