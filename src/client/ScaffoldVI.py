# src/client/ScaffoldVI.py
import torch
from torch.utils.data import DataLoader
from copy import deepcopy
from typing import Dict, Optional, Callable


class ScaffoldVIClient:
    """
    改进 SCAFFOLD（带 VI 正则）的 Client 端：
      - 本地步使用： Fi + eta_r * ∇f̃_i  - c_i + c_global
      - 轮末更新本地控制变量： c_i^r = (1/K) * Σ_k [ Fi + eta_r * ∇f̃_i ]
      - 回传增量： Δy_i = y_{i,K} - x^{r-1} ,  Δc_i = c_i^r - c_i^{r-1}
    说明：
      * Fi: callable(model, **fi_kwargs) -> dict[name -> tensor]，返回与参数同形的“算子梯度”
            若为 None，则视为全 0。
      * eta_r 只作用在“数据梯度”上（∇f̃_i），与我们之前的设定一致。
    """

    def __init__(self, cid, model, dataset, *, lr, batch_size, device):
        self.cid = cid
        self.device = torch.device(device)
        self.model = deepcopy(model).to(self.device)
        self.dataset = dataset
        self.batch_size = batch_size
        self.lr = lr

        # 本地控制变量 c_i（state_dict 形状，初始为 0）
        self.ci: Dict[str, torch.Tensor] = None

        # 用于计算 Δy：保存本轮收到的全局权重 x^{r-1}
        self._x_prev: Dict[str, torch.Tensor] = None
        # 本轮收到的全局控制变量 c^{r-1}
        self._c_global_prev: Dict[str, torch.Tensor] = None
        self._xbar = None

    # ---------- 广播接收 ---------- #
    def set_broadcast(self, global_state: Dict[str, torch.Tensor],
                    global_c: Dict[str, torch.Tensor],
                    xbar_state: Dict[str, torch.Tensor] | None = None) -> None:
        state_on_dev = {k: v.to(self.device) for k, v in global_state.items()}
        self.model.load_state_dict(state_on_dev, strict=True)
        self._x_prev = {k: v.detach().cpu() for k, v in global_state.items()}
        self._c_global_prev = {k: v.detach().to(self.device) for k, v in global_c.items()}
        if self.ci is None:
            self.ci = {k: torch.zeros_like(v, device=self.device) for k, v in state_on_dev.items()}
        # ★ 保存 \bar{x}
        self._xbar = None if xbar_state is None else {k: v.detach().to(self.device) for k, v in xbar_state.items()}

    # 为了兼容你之前的调用习惯，留一个别名
    def set_global(self, global_state, global_c):
        self.set_broadcast(global_state, global_c)

    # ---------- 基础工具 ---------- #
    def get_state(self) -> Dict[str, torch.Tensor]:
        return {k: v.detach().cpu() for k, v in self.model.state_dict().items()}

    def num_samples(self) -> int:
        return len(self.dataset)

    def _loader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)

    # ---------- 训练一轮 ---------- #
    def train_one_round(
        self,
        *,
        local_epochs: int = 1,
        eta_r: float = 0.5,
        Fi: Optional[Callable[..., Dict[str, torch.Tensor]]] = None,
        fi_kwargs: Optional[dict] = None,
        clip_grad: Optional[float] = None,
        lambda_reg: float = 0.1,
    ):
        """
        返回：
          {
            "cid": int,
            "num_samples": int,
            "delta_y": Dict[name->Tensor],   # Δy_i
            "delta_c": Dict[name->Tensor],   # Δc_i
            "ci": Dict[name->Tensor],        # 当前 c_i^r（可选，便于服务器存档）
          }
        """
        assert self._x_prev is not None and self._c_global_prev is not None, \
            "Must call set_broadcast(global_state, global_c) before training."

        if fi_kwargs is None:
            fi_kwargs = {}

        self.model.train()
        opt = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        loss_fn = torch.nn.CrossEntropyLoss()
        loader = self._loader()

        # 累计 G_k = (Fi + eta_r * ∇f̃_i)，用于计算本地 c_i^r = mean_k G_k
        accum_G: Dict[str, torch.Tensor] = {k: torch.zeros_like(p, device=self.device)
                                            for k, p in self.model.state_dict().items()}

        total_steps = 0

        for _ in range(local_epochs):
            for x, y in loader:
                total_steps += 1
                x, y = x.to(self.device), y.to(self.device)
                opt.zero_grad()

                # 1) 数据损失与梯度 ∇f̃_i
                logits = self.model(x)
                loss = loss_fn(logits, y)
                loss.backward()  # param.grad = ∇f̃_i

                # 2) Fi（若提供），与 ∇f̃_i 合成 G = Fi + eta_r * ∇f̃_i
                Fi_grads = {}
                if Fi is not None:
                    # 作为外部算子，不走 autograd 图
                    with torch.no_grad():
                        Fi_grads = Fi(self.model, **fi_kwargs)  # {name: tensor}（在 device 上）

                with torch.no_grad():
                    for name, p in self.model.named_parameters():
                        if p.grad is None:
                            continue
                        g = eta_r * p.grad
                        if name in Fi_grads:
                            g = g + Fi_grads[name].to(self.device)

                        # ★ 加入 toward \bar{x} 的正则梯度 λ (x_i - \bar{x})
                        if (self._xbar is not None) and (lambda_reg > 0.0):
                            g = g + lambda_reg * (p.data - self._xbar[name])

                        # SCAFFOLD 校正（个性化版通常 c_global=0）
                        g = g - self.ci[name] + self._c_global_prev[name]

                        p.grad.copy_(g)

                    if clip_grad is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip_grad)

                opt.step()

        # 3) 本地控制变量更新： c_i^r = (1/K) * accum_G
        K = max(1, total_steps)
        ci_prev = {k: v.detach().clone() for k, v in self.ci.items()}
        with torch.no_grad():
            for name in self.ci.keys():
                self.ci[name] = (accum_G[name] / float(K)).detach().clone()

        # 4) 形成 Δy 与 Δc
        y_state = self.get_state()  # y_{i,K}
        delta_y = {k: (y_state[k] - self._x_prev[k]) for k in y_state.keys()}      # CPU tensors
        delta_c = {k: (self.ci[k].detach().cpu() - ci_prev[k].detach().cpu()) for k in self.ci.keys()}

        return {
            "cid": self.cid,
            "num_samples": self.num_samples(),
            "delta_y": delta_y,
            "delta_c": delta_c,
            "ci": {k: v.detach().cpu() for k, v in self.ci.items()},
        }

    # ---------- 评估 ---------- #
    @torch.no_grad()
    def evaluate(self, dataset, *, batch_size, device, loss_fn):
        self.model.eval()
        device = torch.device(device)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        total_loss, correct, total = 0.0, 0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = self.model(x)
            loss = loss_fn(logits, y)
            total_loss += loss.item() * y.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        return {
            "loss": (total_loss / total) if total > 0 else 0.0,
            "accuracy": (correct / total) if total > 0 else 0.0,
            "num_samples": total
        }
