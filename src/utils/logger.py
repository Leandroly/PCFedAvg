# src/utils/logger.py
import os
import torch
from typing import Dict, Iterable, List

# --------- 客户端本地：把自有参数拼成向量 + 生成一步的记录 ---------
def vectorize_owned(state_dict: Dict[str, torch.Tensor], owned_keys: Iterable[str]) -> torch.Tensor:
    vecs = []
    for k in owned_keys:
        if k in state_dict:
            v = state_dict[k]
            vecs.append(v.detach().view(-1).cpu())
    return torch.cat(vecs) if vecs else torch.empty(0)

def make_trace_entry(step_id: int, vec: torch.Tensor, max_show: int = 5) -> Dict:
    if vec.numel() == 0:
        return {}
    head = [float(x) for x in vec[:max_show].tolist()]
    return {"k": int(step_id), "norm": float(vec.norm().item()), "head": head}


# --------- 服务器侧：轮次级别的记录器 ---------
class TraceLogger:
    def __init__(self, log_file: str = "fedvi_trace.log", overwrite: bool = True):
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        self.log_file = log_file
        if overwrite:
            with open(self.log_file, "w", encoding="utf-8") as f:
                f.write("=== FedVI Trace Log ===\n")

    # 只写文件
    def _write(self, msg: str):
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    # 终端 + 文件（摘要用）
    def _log(self, msg: str):
        print(msg)
        self._write(msg)

    # 只写文件（矩阵用）
    def _log_file_only(self, msg: str):
        self._write(msg)

    def print_summary(self, cid: int, trace: List[Dict], step_count: int | None = None):
        if not trace:
            steps = 0 if step_count is None else int(step_count)
            self._log(f"[TRACE] cid={cid} | steps={steps} | norm: - -> - | head0: [] -> headK: []")
            return
        first, last = trace[0], trace[-1]
        steps = int(step_count) if step_count is not None else len(trace)
        self._log(
            f"[TRACE] cid={cid} | steps={steps} | "
            f"norm: {first['norm']:.6f} -> {last['norm']:.6f} | "
            f"head0: {first['head']} -> headK: {last['head']}"
        )

    # —— 矩阵视图：只写入文件，不在终端打印 ——
    def write_matrix_style(self, round_id: int, traces_by_client: Dict[int, List[Dict]],
                           value: str = "norm", orientation: str = "rows=steps"):
        self._log_file_only(f"[MATRIX] round={round_id} value={value} orientation={orientation}")
        cids = sorted(traces_by_client.keys())
        max_len = max((len(traces_by_client[c]) for c in cids), default=0)

        def get_val(e):
            return e.get(value, None)

        if orientation == "rows=steps":
            header = "step ".ljust(6) + " | " + " | ".join([f"cid={c}".center(12) for c in cids])
            sep = "-" * len(header)
            self._log_file_only(header)
            self._log_file_only(sep)
            for k in range(1, max_len + 1):
                row_vals = []
                for c in cids:
                    tr = traces_by_client[c]
                    if k <= len(tr):
                        v = get_val(tr[k - 1])
                        cell = f"{v:.6f}" if isinstance(v, float) else str(v)
                    else:
                        cell = ""
                    row_vals.append(cell.rjust(12))
                self._log_file_only(f"{str(k).rjust(4)}  | " + " | ".join(row_vals))
        else:  # rows=clients
            header = "client ".ljust(8) + " | " + " | ".join([f"k={k}".center(12) for k in range(1, max_len + 1)])
            sep = "-" * len(header)
            self._log_file_only(header)
            self._log_file_only(sep)
            for c in cids:
                tr = traces_by_client[c]
                row_vals = []
                for k in range(1, max_len + 1):
                    if k <= len(tr):
                        v = get_val(tr[k - 1])
                        cell = f"{v:.6f}" if isinstance(v, float) else str(v)
                    else:
                        cell = ""
                    row_vals.append(cell.rjust(12))
                self._log_file_only(f"{str(c).rjust(6)}  | " + " | ".join(row_vals))

    def log_round(self, round_id: int, payloads: List[Dict], *,
              all_client_ids: List[int], vec_dim: int,
              baselines: Dict[int, Dict], step_count: int):
        """
        baselines[cid] = {"norm": float, "head": List[float]}
        终端：打印形状、Selected/Not selected 列表 + 两类 clients 的摘要 TRACE
        文件：写入包含所有 clients 的矩阵（未选中的用“静止轨迹”补全）
        """
        selected = sorted(p["cid"] for p in payloads)
        all_cids = sorted(all_client_ids)
        not_selected = [c for c in all_cids if c not in selected]

        m, n = len(all_cids), int(vec_dim)

        self._log(f"=== Round {round_id} Trace Summary ===")
        self._log(f"[Y shape] m x n = {m} x {n}")
        self._log(f"[Selected] {selected}")
        self._log(f"[Not selected] {not_selected}")

        # 已选中的：真实 trace
        traces_by_client = {p["cid"]: p.get("trace", []) for p in payloads}
        for cid in selected:
            self.print_summary(cid, traces_by_client.get(cid, []), step_count = step_count)

        # 未选中的：打印摘要（steps = step_count，首尾相同）
        for cid in not_selected:
            b = baselines.get(cid, None)
            if b is None:
                continue
            a = b["norm"]
            head = b["head"]
            self._log(
                f"[TRACE] cid={cid} | steps={step_count} | norm: {a:.6f} -> {a:.6f} | "
                f"head0: {head} -> headK: {head}"
            )

        self._log(f"=== End Round {round_id} ===")

        # ===== 仅写文件：矩阵（包含所有 clients；未选中的用静止轨迹补齐）=====
        # 先复制真实的，再为未选中的合成静止轨迹
        full_traces = dict(traces_by_client)
        for cid in not_selected:
            b = baselines.get(cid, None)
            if b is None:
                continue
            v = b["norm"]
            head = b["head"]
            # 合成长度为 step_count 的“静止轨迹”
            full_traces[cid] = [
                {"k": k, "norm": float(v), "head": head} for k in range(1, step_count + 1)
            ]

        # 写入矩阵（行=steps，列=所有 clients）
        self.write_matrix_style(
            round_id,
            full_traces,
            value="norm",
            orientation="rows=steps",
        )