# src/client/ScaffoldVI.py
import torch
from torch.utils.data import DataLoader
from copy import deepcopy
from typing import Dict, Optional


class ScaffoldVIClient:
    def __init__(self, cid, model, dataset, *, lr, batch_size, device, owned_keys):
        self.cid = cid
        self.device = torch.device(device)
        self.model = deepcopy(model).to(self.device)
        self.dataset = dataset
        self.batch_size = batch_size
        self.lr = float(lr)

        # 仅训练本客户端拥有的 block
        self.owned_keys = set(owned_keys)
        for n, p in self.model.named_parameters():
            p.requires_grad = (n in self.owned_keys)

        # SCAFFOLD 变量 & 缓存
        self.ci: Dict[str, torch.Tensor] = None                   # 仅 owned_keys
        self._x_prev: Dict[str, torch.Tensor] = None              # 仅 owned_keys（CPU）
        self._c_global_prev: Dict[str, torch.Tensor] = None       # 仅 owned_keys（device）
        self._xbar: Optional[Dict[str, torch.Tensor]] = None      # 仅 owned_keys（device）

    # ===== 与服务器交互 =====
    def set_broadcast(
        self,
        global_state: Dict[str, torch.Tensor],
        global_c: Dict[str, torch.Tensor],
        xbar_state: Optional[Dict[str, torch.Tensor]] = None
    ) -> None:
        """加载（或刷新）广播的全局模型/全局控制变元，并可选提供 \bar{x}。"""
        # 加载参数到本地模型（全量同结构，正常 strict=True）
        state_on_dev = {k: v.to(self.device) for k, v in global_state.items()}
        self.model.load_state_dict(state_on_dev, strict=True)

        # 只保存 owned_keys 的快照
        self._x_prev = {k: v.detach().cpu() for k, v in global_state.items() if k in self.owned_keys}
        self._c_global_prev = {k: v.detach().to(self.device) for k, v in global_c.items() if k in self.owned_keys}

        # 初始化本地 c_i（仅 owned_keys）
        if self.ci is None:
            self.ci = {k: torch.zeros_like(state_on_dev[k], device=self.device) for k in self.owned_keys}

        # 可选：\bar{x} 也只保留 owned_keys
        if xbar_state is None:
            self._xbar = None
        else:
            self._xbar = {k: v.detach().to(self.device) for k, v in xbar_state.items() if k in self.owned_keys}

    # 兼容旧名
    def set_global(self, global_state, global_c):
        self.set_broadcast(global_state, global_c, None)

    # ===== 基础工具 =====
    def get_state(self) -> Dict[str, torch.Tensor]:
        sd = self.model.state_dict()
        return {k: v.detach().cpu() for k, v in sd.items() if k in self.owned_keys}

    def num_samples(self) -> int:
        return len(self.dataset)

    def _loader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)

    # ===== 训练一轮 =====
    def train_one_round(
        self,
        *,
        local_epochs: int = 1,
        eta_r: float = 0.5,
        lambda_reg: float = 0.1,
        clip_grad: Optional[float] = None,
    ):
        assert self._x_prev is not None and self._c_global_prev is not None, \
            "Must call set_broadcast(global_state, global_c) before training."

        self.model.train()
        opt = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr)
        loss_fn = torch.nn.CrossEntropyLoss()
        loader = self._loader()

        # 保存 w0（仅 owned_keys，设备上）
        w0_dev = {n: p.detach().clone() for n, p in self.model.named_parameters() if n in self.owned_keys}

        total_steps = 0
        for _ in range(local_epochs):
            for x, y in loader:
                total_steps += 1
                x, y = x.to(self.device), y.to(self.device)
                opt.zero_grad()

                # 1) ∇f̃_i
                logits = self.model(x)
                loss = loss_fn(logits, y)
                loss.backward()  # p.grad = ∇f̃_i

                # 2) 合成梯度： g = F_i + η_r * ∇f̃_i - c_i + c_global
                with torch.no_grad():
                    for name, p in self.model.named_parameters():
                        if (name not in self.owned_keys) or (p.grad is None):
                            continue

                        # F_i = ∇f̃_i + λ (x - \bar{x})（若提供了 \bar{x}）
                        Fi = p.grad.clone()
                        if (lambda_reg > 0.0) and (self._xbar is not None) and (name in self._xbar):
                            Fi.add_(lambda_reg * (p.data - self._xbar[name]))

                        g = Fi + eta_r * p.grad - self.ci[name] + self._c_global_prev[name]
                        p.grad.copy_(g)

                    if clip_grad is not None:
                        torch.nn.utils.clip_grad_norm_(
                            filter(lambda p: p.requires_grad, self.model.parameters()),
                            max_norm=clip_grad
                        )

                opt.step()

        # 3) 用参数差分更新 c_i： c_i^{new} = c_i^{old} - c + (w0 - wK)/(K*lr)
        K = max(1, total_steps)
        ci_prev = {k: v.detach().clone() for k, v in self.ci.items()}
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                if name not in self.owned_keys:
                    continue
                diff = w0_dev[name] - p.detach()            # w0 - wK
                self.ci[name] = ci_prev[name] - self._c_global_prev[name] + diff / (K * self.lr)

        # 4) 组装增量（只含 owned_keys）
        y_state = self.get_state()  # CPU
        delta_y = {k: (y_state[k] - self._x_prev[k]) for k in y_state.keys()}
        delta_c = {k: (self.ci[k].detach().cpu() - ci_prev[k].detach().cpu()) for k in self.ci.keys()}

        return {
            "cid": self.cid,
            "num_samples": self.num_samples(),
            "delta_y": delta_y,
            "delta_c": delta_c,
            "ci": {k: v.detach().cpu() for k, v in self.ci.items()},
        }

    # ===== 评估 =====
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
