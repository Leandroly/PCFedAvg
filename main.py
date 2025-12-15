# main_fedvi_avg.py
import torch, random, numpy as np
import matplotlib.pyplot as plt
import copy
import math

from src.utils.config import DATASET, MODEL, TRAINING, OPTIMIZER, LOSS_FN
from src.utils.models import OneNN
from src.client.FedVI import FedVIClient
from src.server.FedVI import FedVIServer
from generate_data import mnist_subsets

SAMPLE_ROUNDS = [0, 10, 20, 40, 60, 80, 100]
KS = [1, 10, 20]
REPEATS = 2


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _eta_schedule(r: int) -> float:
    # 目前先固定 0.2，你以后可以根据 round 调整
    return 0.2


def _print_all_blocks(clients, max_show: int = 5):
    for c in clients:
        state = c.get_block_state()
        print(f"---- Client[{c.cid}] block snapshot ----")
        for name, t in state.items():
            flat = t.view(-1).float()
            head = ", ".join([f"{v:.4f}" for v in flat[:max_show].tolist()])
            tail = " ..." if flat.numel() > max_show else ""
            print(f"{name:20s} shape={list(t.shape)!r}  values=[{head}{tail}]")
        print("---------------------------------------------------")


def _build_server_and_clients(init_state, train_subsets, device, overwrite=False):
    """小工具：给定初始权重和数据，构造一套 global_model + clients + server。"""
    global_model = OneNN(in_dim=MODEL["in_dim"], num_classes=MODEL["num_classes"])
    global_model.load_state_dict(copy.deepcopy(init_state))

    owned_keys = list(global_model.state_dict().keys())

    clients = [
        FedVIClient(
            cid=i,
            model=global_model,
            dataset=train_subsets[i],
            lr=OPTIMIZER["lr"],
            batch_size=TRAINING["train_batch_size"],
            device=device,
            owned_keys=owned_keys,
            m_total=DATASET["num_clients"],
        )
        for i in range(DATASET["num_clients"])
    ]
    server = FedVIServer(global_model, clients, device=device, overwrite=overwrite)
    return server, clients


def calibrate_rho_for_rep(init_state, train_subsets, device):
    """
    用当前 rep 的数据和初始模型做一次 rho 校准：
    - 构造一套 server/clients
    - 调用 server.calibrate_rho() 做若干轮 warmup
    - 返回 rho_mid 等信息
    """
    server, _ = _build_server_and_clients(init_state, train_subsets, device, overwrite=False)

    # 这里的 warmup 超参数你可以改：local_steps、fraction、eta_r
    stats = server.calibrate_rho(
        warmup_rounds=2,
        fraction=TRAINING["fraction"],
        local_steps=1,
        eta_r=0.2,   # warmup 时不加额外 eta 扰动
    )
    print(
        f"[Calibrate rho] median_drift={stats['median_drift']:.4f}, "
        f"rho_tight={stats['rho_tight']:.4f}, "
        f"rho_mid={stats['rho_mid']:.4f}, "
        f"rho_loose={stats['rho_loose']:.4f}"
    )
    return stats


def run_fedvi_with_k(k_value, init_state, train_subsets, testset, device, rep, rho_mid: float, overwrite=False):
    # 用统一的 rho_mid 构造一套新的 server + clients，从相同 init_state 开始训练
    server, clients = _build_server_and_clients(init_state, train_subsets, device, overwrite=overwrite)
    drifts_over_rounds = {0: server.compute_drifts()}
    server.log_round0()
    server.logger._write(f"\n=== START EXPERIMENT === k={k_value}, rep={rep} ===\n")

    # ---------- Global Round-0 ----------
    metrics = server.evaluate_global(
        dataset=testset, batch_size=TRAINING["eval_batch_size"],
        device=device, loss_fn=LOSS_FN,
    )
    losses = {0: metrics["loss"]}

    # ---------- Local Round-0 ----------
    locals_metrics_0 = server.evaluate_locals(
        batch_size=TRAINING["eval_batch_size"],
        device=device,
        loss_fn=LOSS_FN,
    )
    local_losses_by_client = {m["cid"]: {0: m["loss"]} for m in locals_metrics_0}

    print(f"[FedVI k={k_value}, Round 0] acc={metrics['accuracy']:.4f} loss={metrics['loss']:.4f}")

    # ---------- 训练循环 ----------
    for r in range(1, TRAINING["rounds"] + 1):
        eta_r = _eta_schedule(r)
        stats = server.run_round(
            fraction=TRAINING["fraction"],
            local_steps=k_value,
            eta_r=eta_r,
            rho=rho_mid,   # 这里用 rho_mid
        )
        print(f"[k={k_value}, Round {r}] selected={stats['selected']} | "
              f"total_samples={stats['total_samples']} | eta={eta_r:.3f} | rho_mid={rho_mid:.3f}")
        
        drifts_over_rounds[r] = server.compute_drifts()

        if r % 10 == 0:
            _print_all_blocks(clients, max_show=5)

        metrics = server.evaluate_global(
            dataset=testset, batch_size=TRAINING["eval_batch_size"],
            device=device, loss_fn=LOSS_FN,
        )
        if r in SAMPLE_ROUNDS:
            losses[r] = metrics["loss"]

        # 在采样轮记录“每个 client 的 local loss”
        if r in SAMPLE_ROUNDS:
            locals_metrics = server.evaluate_locals(
                batch_size=TRAINING["eval_batch_size"],
                device=device,
                loss_fn=LOSS_FN,
            )
            for m in locals_metrics:
                cid = m["cid"]
                if cid not in local_losses_by_client:
                    local_losses_by_client[cid] = {}
                local_losses_by_client[cid][r] = m["loss"]

        if (r % 10 == 0) or (r == TRAINING["rounds"]):
            print(f"[FedVI k={k_value}, Round {r}] acc={metrics['accuracy']:.4f} loss={metrics['loss']:.4f}")

    # ---------- 返回：全局曲线 + 各 client 曲线 ----------
    global_curve = [losses[r] for r in SAMPLE_ROUNDS]
    local_curves_by_client = {
        cid: [local_losses_by_client[cid].get(r, float('nan')) for r in SAMPLE_ROUNDS]
        for cid in range(DATASET["num_clients"])
    }
    return global_curve, local_curves_by_client, drifts_over_rounds


def main():
    device = TRAINING["device"]
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    print(f"Device: {device}")

    # 这些 dict 用来存所有 rep 和 k 的曲线
    all_results_global = {k: [] for k in KS}
    all_results_local_by_client = {k: {cid: [] for cid in range(DATASET["num_clients"])} for k in KS}
    all_results_drifts = {k: [] for k in KS}

    for rep in range(REPEATS):
        print(f"\n===== Repeat {rep+1}/{REPEATS} =====")
        seed = TRAINING["seed"] + rep
        set_seed(seed)

        train_subsets, testset = mnist_subsets(
            n_clients=DATASET["num_clients"],
            scheme=DATASET["partition"],
            alpha=DATASET["dirichlet_alpha"],
            seed=seed,
            root=DATASET["root"],
        )

        base_model = OneNN(in_dim=MODEL["in_dim"], num_classes=MODEL["num_classes"])
        init_state = copy.deepcopy(base_model.state_dict())

        # 先用当前数据和初始模型校准一次 rho（只调一次，所有 k 共用）
        rho_stats = calibrate_rho_for_rep(init_state, train_subsets, device)
        rho_mid = rho_stats["rho_mid"]

        for k in KS:
            overwrite_flag = (rep == 0 and k == KS[0])
            curve_global, local_curves_by_client, drifts_over_rounds = run_fedvi_with_k(
                k, init_state, train_subsets, testset, device,
                rep=rep+1, rho_mid=rho_mid, overwrite=overwrite_flag
            )
            all_results_global[k].append(curve_global)
            for cid, curve_local in local_curves_by_client.items():
                all_results_local_by_client[k][cid].append(curve_local)
            all_results_drifts[k].append(drifts_over_rounds)

    # ============================================================
    # (1) Figure 1: 全局 Loss
    # ============================================================
    plt.figure(figsize=(8, 6))
    for k in KS:
        arr = np.array(all_results_global[k])
        mean_curve = arr.mean(axis=0)
        plt.plot(SAMPLE_ROUNDS, mean_curve, marker="o", label=f"k={k}")

    plt.xlabel("Round")
    plt.ylabel("Global Loss")
    plt.title("FedVI Global Loss vs Round")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()



    # ============================================================
    # (2) Figure 2: 每个 client 的 local loss（20 个子图）
    # ============================================================
    num_clients = DATASET["num_clients"]
    rows = int(math.ceil(num_clients / 5))
    cols = 5

    fig, axes = plt.subplots(rows, cols, figsize=(16, 3.2 * rows),
                            sharex=True, sharey=True)

    # 统一 axes 成 2D
    if rows == 1:
        axes = [axes] if cols > 1 else [[axes]]

    for cid in range(num_clients):
        r, c = divmod(cid, cols)
        ax = axes[r][c]

        for k in KS:
            curves = np.array(all_results_local_by_client[k][cid])
            mean_curve = curves.mean(axis=0)
            ax.plot(SAMPLE_ROUNDS, mean_curve, marker="o", label=f"k={k}")

        ax.set_title(f"Client {cid}")
        ax.grid(True)
        if r == rows - 1:
            ax.set_xlabel("Round")
        if c == 0:
            ax.set_ylabel("Local Loss")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(KS))
    fig.suptitle("FedVI - Local Loss per Client (mean over repeats)")
    plt.tight_layout(rect=(0, 0.05, 1, 0.95))
    plt.show()



    # ============================================================
    # (3) Figure 3: 全局平均 drift
    # ============================================================
    plt.figure(figsize=(8, 6))
    for k in KS:
        curves = []
        for drifts_over_rounds in all_results_drifts[k]:
            avg_drift = [
                np.mean(list(drifts_over_rounds[r].values()))
                for r in SAMPLE_ROUNDS
            ]
            curves.append(avg_drift)
        mean_curve = np.mean(curves, axis=0)
        plt.plot(SAMPLE_ROUNDS, mean_curve, marker="o", label=f"k={k}")

    plt.xlabel("Round")
    plt.ylabel(r"Average Drift $\|x_i - \bar{x}\|$")
    plt.title("FedVI Global Drift vs Round")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()



    # ============================================================
    # (4) Figure 4: 每个 client 的 drift（20 个子图）
    # ============================================================
    fig, axes = plt.subplots(rows, cols, figsize=(16, 3.2 * rows),
                            sharex=True, sharey=True)

    # 统一 axes 成 2D
    if rows == 1:
        axes = [axes] if cols > 1 else [[axes]]

    for cid in range(num_clients):
        r, c = divmod(cid, cols)
        ax = axes[r][c]

        for k in KS:
            curves = []
            for drifts_over_rounds in all_results_drifts[k]:
                curve = [drifts_over_rounds[r_round][cid] for r_round in SAMPLE_ROUNDS]
                curves.append(curve)
            mean_curve = np.mean(curves, axis=0)
            ax.plot(SAMPLE_ROUNDS, mean_curve, marker="o", label=f"k={k}")

        ax.set_title(f"Client {cid} Drift")
        ax.grid(True)
        if r == rows - 1:
            ax.set_xlabel("Round")
        if c == 0:
            ax.set_ylabel(r"Drift $\|x_i - \bar{x}\|$")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(KS))
    fig.suptitle("FedVI - Drift per Client (mean over repeats)")
    plt.tight_layout(rect=(0, 0.05, 1, 0.95))
    plt.show()





if __name__ == "__main__":
    main()
