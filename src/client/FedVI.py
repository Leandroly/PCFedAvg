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

        # 只让本 client 自己的 block 参与梯度更新
        for n, p in self.model.named_parameters():
            p.requires_grad = (n in self.owned_keys)

        # \bar{x}^{r-1}（上一轮 server 的块平均）
        self._xbar_prev: Dict[str, torch.Tensor] = {}
        # 本 client 上一轮自己的块 x^{r-1,i}
        self._x_prev_block: Dict[str, torch.Tensor] = {}
        # 其它 client 的块（如果需要的话）
        self._others_prev_blocks: List[Dict[str, torch.Tensor]] | None = None

    def set_block_weights(
        self,
        xbar_prev: Dict[str, torch.Tensor],
        x_prev_block: Dict[str, torch.Tensor],
        others_prev_blocks: List[Dict[str, torch.Tensor]] | None = None,
    ):
        # 存上一轮 server 端的块平均 \bar{x}^{r-1}
        self._xbar_prev = {k: v.detach().clone().to(self.device) for k, v in xbar_prev.items()}
        # 存本 client 上一轮自己的块 x^{r-1,i}
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
    rho: float = 1.0,       # 约束阈值 ρ
    ):
        """
        本地一轮训练：

            g_{i,t}^{r,η_r} = ∇_x g_i(x_{i,t}) + η_r ∇_x f̃_i(\bar x^{r-1}, ξ_{i,t})

        其中
        f̃_i: 本地交叉熵损失
        h_i(x_i) = ||x_i - x_hat_local||_2
        g_i(x_i) = 0.5 * max(0, h_i(x_i) - rho)^2
        x_hat_local = \bar{x}^{r-1} + (x_i - x_i^{r-1}) / m   （块的“动态平均”）
        """
        trace = []
        vec_dim = 0
        self.model.train()
        opt = torch.optim.SGD(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.lr
        )
        loss_fn = torch.nn.CrossEntropyLoss()
        loader = self._loader()

        m = self.m_total
        it = cycle(loader)

        # k = 0: 初始向量记录
        sd0 = self.model.state_dict()
        y_vec0 = vectorize_owned(sd0, self.owned_keys)
        vec_dim = int(y_vec0.numel())
        rec0 = make_trace_entry(0, y_vec0, max_show=5)
        if rec0:
            trace.append(rec0)

        eps = 1e-12  # 防止除零

        # 预先把 \bar{x}^{r-1} 搬到 device 上，后面反复用
        xbar_prev_dev = {k: v.to(self.device) for k, v in self._xbar_prev.items()}

        for step in range(local_steps):
            x, y = next(it)
            x, y = x.to(self.device), y.to(self.device)

            # ===== 1) 在 \bar{x}^{r-1} 上计算 ∇f̃_i(\bar x^{r-1}, ξ) =====
            # 先保存当前本地参数 x_i
            local_state = {
                k: v.detach().clone()
                for k, v in self.model.state_dict().items()
            }

            # 加载 \bar{x}^{r-1} 作为当前模型参数
            self.model.load_state_dict(xbar_prev_dev, strict=False)

            opt.zero_grad()
            logits_bar = self.model(x)
            loss_bar = loss_fn(logits_bar, y)
            loss_bar.backward()

            # 把在 \bar x 上的梯度保存下来
            grad_f_bar: Dict[str, torch.Tensor] = {}
            for name, p in self.model.named_parameters():
                if name in self.owned_keys and p.grad is not None:
                    grad_f_bar[name] = p.grad.detach().clone()

            # ===== 2) 恢复本地参数 x_i，计算 ∇g_i(x_i) =====
            self.model.load_state_dict(local_state, strict=False)

            with torch.no_grad():
                diffs: Dict[str, torch.Tensor] = {}
                sq_norm = torch.zeros(1, device=self.device)

                # 先算 h_i(x_i) = ||x_i - x_hat_local||，注意 x_hat_local 依赖 x_i^{r-1}
                for name, p in self.model.named_parameters():
                    if not (p.requires_grad and name in self.owned_keys):
                        continue

                    xbar_local = self._xbar_prev[name] + (p.data - self._x_prev_block[name]) / float(m)
                    diff = p.data - xbar_local
                    diffs[name] = diff
                    sq_norm += (diff ** 2).sum()

                h_val = torch.sqrt(sq_norm + eps)
                violation = torch.clamp(h_val - rho, min=0.0)

            # ===== 3) 组合梯度：∇g_i(x_i) + η_r ∇f̃_i(\bar x) =====
            opt.zero_grad()
            for name, p in self.model.named_parameters():
                if not (p.requires_grad and name in self.owned_keys):
                    continue

                # ∇f̃_i(\bar x) 只在 owned_keys 上有
                grad_ftilde_bar = grad_f_bar.get(name, torch.zeros_like(p.data))

                # ∇g_i(x_i)
                if violation.item() > 0.0:
                    diff = diffs[name]
                    grad_g = violation * diff / (h_val + eps)
                else:
                    grad_g = torch.zeros_like(p.data)

                total_grad = grad_g + eta_r * grad_ftilde_bar
                p.grad = total_grad

            opt.step()

            # ===== 4) 记录参数轨迹 =====
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

