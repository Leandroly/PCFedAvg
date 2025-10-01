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

SAMPLE_ROUNDS = [0, 10, 20, 30, 40, 50]
KS = [1, 5, 10]
REPEATS = 2

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _eta_schedule(r: int) -> float:
    return 0.2


def _lambda_schedule(r: int) -> float:
    return 0.1


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


def run_fedvi_with_k(k_value, init_state, train_subsets, testset, device, rep, overwrite=False):
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
    server.log_round0()
    server.logger._write(f"\n=== START EXPERIMENT === k={k_value}, rep={rep} ===\n")

    # ---------- Global Round-0 ----------
    metrics = server.evaluate_global(
        dataset=testset, batch_size=TRAINING["eval_batch_size"],
        device=device, loss_fn=LOSS_FN,
    )
    losses = {0: metrics["loss"]}

    # ---------- Local Round-0（关键：在任何训练前记录，确保不同 k 起点一致） ----------
    locals_metrics_0 = server.evaluate_locals(
        batch_size=TRAINING["eval_batch_size"],
        device=device,
        loss_fn=LOSS_FN,
    )
    # 对每个 client 建立曲线容器，并放入 round=0 的值
    local_losses_by_client = {m["cid"]: {0: m["loss"]} for m in locals_metrics_0}

    print(f"[FedVI k={k_value}, Round 0] acc={metrics['accuracy']:.4f} loss={metrics['loss']:.4f}")

    # ---------- 训练循环 ----------
    for r in range(1, TRAINING["rounds"] + 1):
        eta_r = _eta_schedule(r)
        lam_r = _lambda_schedule(r)

        stats = server.run_round(
            fraction=TRAINING["fraction"],
            local_steps=k_value,
            eta_r=eta_r,
            lambda_reg=lam_r,
        )
        print(f"[k={k_value}, Round {r}] selected={stats['selected']} | "
              f"total_samples={stats['total_samples']} | eta={eta_r:.3f} | lambda={lam_r:.3f}")

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
    return global_curve, local_curves_by_client

def main():
    device = TRAINING["device"]
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    print(f"Device: {device}")

    all_results = {k: [] for k in KS}
    all_results_global = {k: [] for k in KS}
    all_results_local_by_client = {k: {cid: [] for cid in range(DATASET["num_clients"])} for k in KS}

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

        for k in KS:
            overwrite_flag = (rep == 0 and k == KS[0])
            curve = run_fedvi_with_k(k, init_state, train_subsets, testset, device, rep=rep+1, overwrite=overwrite_flag)
            all_results[k].append(curve)
            curve_global, local_curves_by_client = run_fedvi_with_k(
                k, init_state, train_subsets, testset, device, rep=rep+1, overwrite=overwrite_flag
            )
            all_results_global[k].append(curve_global)
            for cid, curve_local in local_curves_by_client.items():
                all_results_local_by_client[k][cid].append(curve_local)

    plt.figure(figsize=(8, 6))
    for k in KS:
        arr = np.array(all_results_global[k])
        mean = arr.mean(axis=0)
        plt.plot(SAMPLE_ROUNDS, mean, marker="o", label=f"k={k}")

    plt.xlabel("Round")
    plt.ylabel("Loss")
    plt.title("FedVI Global Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    num_clients = DATASET["num_clients"]
    rows = int(math.ceil(num_clients / 5))
    cols = 5 if num_clients >= 5 else num_clients
    fig, axes = plt.subplots(rows, cols, figsize=(16, 3.2*rows), sharex=True, sharey=True)
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]

    for cid in range(num_clients):
        r, c = divmod(cid, cols)
        ax = axes[r][c]
        for k in KS:
            curves = np.array(all_results_local_by_client[k][cid])  # shape: [REPEATS, len(SAMPLE_ROUNDS)]
            mean_curve = curves.mean(axis=0) if curves.size > 0 else np.full(len(SAMPLE_ROUNDS), np.nan)
            ax.plot(SAMPLE_ROUNDS, mean_curve, marker="o", label=f"k={k}")
            ax.set_title(f"Client {cid}")
        if r == rows - 1:
            ax.set_xlabel("Round")
        if c == 0:
            ax.set_ylabel("Local Loss")
        ax.grid(True)

    # 统一图例（放在下方）
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(KS))
    fig.suptitle("FedVI - Local Loss per Client (mean over repeats)")
    plt.tight_layout(rect=(0, 0.05, 1, 0.95))
    plt.show()

if __name__ == "__main__":
    main()
