# main_fedvi_avg.py
import torch, random, numpy as np
import matplotlib.pyplot as plt
import copy

from src.utils.config import DATASET, MODEL, TRAINING, OPTIMIZER, LOSS_FN
from src.utils.models import OneNN
from src.client.FedVI import FedVIClient
from src.server.FedVI import FedVIServer
from generate_data import mnist_subsets

SAMPLE_ROUNDS = [0, 10, 20, 30, 40, 50]
KS = [1, 5, 10]
REPEATS = 2  # 重复次数


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _eta_schedule(r: int) -> float:
    # 数据梯度前的系数；可按需改为递减/恒定
    return max(0.1, 0.6 / (r ** 0.5))


def _lambda_schedule(r: int) -> float:
    # ★ 固定 λ = 0.1（来自你的要求）
    return 0.1


def _print_block(block_state: dict, max_show: int = 5):
    print("---- Client[0] block snapshot ----")
    for name, t in block_state.items():
        flat = t.view(-1).float()
        head = ", ".join([f"{v:.4f}" for v in flat[:max_show].tolist()])
        tail = " ..." if flat.numel() > max_show else ""
        print(f"{name:20s} shape={list(t.shape)!r}  values=[{head}{tail}]")
    print("-----------------------------------")


def run_fedvi_with_k(k_value, init_state, train_subsets, testset, device):
    # 初始化全局模型 & 客户端
    global_model = OneNN(in_dim=MODEL["in_dim"], num_classes=MODEL["num_classes"])
    global_model.load_state_dict(copy.deepcopy(init_state))

    # 这里默认每个 client 的 block = 整个模型
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
        )
        for i in range(DATASET["num_clients"])
    ]
    server = FedVIServer(global_model, clients, device=device)

    # baseline r=0（评估全局模型）
    metrics = server.evaluate_global(
        dataset=testset, batch_size=TRAINING["eval_batch_size"],
        device=device, loss_fn=LOSS_FN,
    )
    losses = {0: metrics["loss"]}
    print(f"[FedVI k={k_value}, Round 0] acc={metrics['accuracy']:.4f} loss={metrics['loss']:.4f}")

    for r in range(1, TRAINING["rounds"] + 1):
        eta_r = _eta_schedule(r)
        lam_r = _lambda_schedule(r)

        stats = server.run_round(
            fraction=TRAINING["fraction"],
            local_epochs=k_value,
            eta_r=eta_r,
            lambda_reg=lam_r,     # ★ 传入 λ
            prox_cfg=None,        # 需要其它 Fi（如 FedProx/自定义 VI）时在此传
        )
        print(f"[k={k_value}, Round {r}] selected={stats['selected']} | "
              f"total_samples={stats['total_samples']} | eta={eta_r:.3f} | "
              f"lambda={lam_r:.3f} | RegSum={stats['reg_value']:.3e}")

        # 每 10 轮打印第一个 client 的 block
        if r % 10 == 0:
            block0 = clients[0].get_block_state()
            _print_block(block0, max_show=5)

        # 评估全局模型
        metrics = server.evaluate_global(
            dataset=testset, batch_size=TRAINING["eval_batch_size"],
            device=device, loss_fn=LOSS_FN,
        )
        if r in SAMPLE_ROUNDS:
            losses[r] = metrics["loss"]

        if (r % 10 == 0) or (r == TRAINING["rounds"]):
            print(f"[FedVI k={k_value}, Round {r}] acc={metrics['accuracy']:.4f} loss={metrics['loss']:.4f}")

    return [losses[r] for r in SAMPLE_ROUNDS]


def main():
    device = TRAINING["device"]
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    print(f"Device: {device}")

    all_results = {k: [] for k in KS}

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
            curve = run_fedvi_with_k(k, init_state, train_subsets, testset, device)
            all_results[k].append(curve)

    # —— 画均值 —— #
    plt.figure(figsize=(8, 6))
    for k in KS:
        arr = np.array(all_results[k])  # [REPEATS, 6]
        mean = arr.mean(axis=0)
        plt.plot(SAMPLE_ROUNDS, mean, marker="o", label=f"k={k}")

    plt.xlabel("Round")
    plt.ylabel("Loss")
    plt.title("FedVI (VI-regularized FedAvg)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
