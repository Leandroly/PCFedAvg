# main_scaffold_avg.py
import torch, random, numpy as np
import matplotlib.pyplot as plt
import copy

from src.utils.config import DATASET, MODEL, TRAINING, OPTIMIZER, LOSS_FN
from src.utils.models import OneNN
from src.client.Scaffold import ScaffoldClient
from src.server.Scaffold import ScaffoldServer
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


def run_scaffold_with_k(k_value, init_state, train_subsets, testset, device):
    global_model = OneNN(in_dim=MODEL["in_dim"], num_classes=MODEL["num_classes"])
    global_model.load_state_dict(copy.deepcopy(init_state))

    clients = [
        ScaffoldClient(
            cid=i,
            model=global_model,
            dataset=train_subsets[i],
            lr=OPTIMIZER["lr"],
            batch_size=TRAINING["train_batch_size"],
            device=device,
        )
        for i in range(DATASET["num_clients"])
    ]
    server = ScaffoldServer(global_model, clients, device=device)

    # baseline r=0
    metrics = server.evaluate_global(
        dataset=testset, batch_size=TRAINING["eval_batch_size"],
        device=device, loss_fn=LOSS_FN,
    )
    losses = {0: metrics["loss"]}
    print(f"[SCAFFOLD k={k_value}, Round 0] acc={metrics['accuracy']:.4f} loss={metrics['loss']:.4f}")

    for r in range(1, TRAINING["rounds"] + 1):
        stats = server.run_round(fraction=TRAINING["fraction"], local_epochs=k_value)
        print(f"[k={k_value}, Round {r}] selected={stats['selected']} | total_samples={stats['total_samples']}")

        metrics = server.evaluate_global(
            dataset=testset, batch_size=TRAINING["eval_batch_size"],
            device=device, loss_fn=LOSS_FN,
        )
        if r in SAMPLE_ROUNDS:
            losses[r] = metrics["loss"]

        if (r % 10 == 0) or (r == TRAINING["rounds"]):
            print(f"[SCAFFOLD k={k_value}, Round {r}] acc={metrics['accuracy']:.4f} loss={metrics['loss']:.4f}")

    return [losses[r] for r in SAMPLE_ROUNDS]  # 只返回 6 个点


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
            curve = run_scaffold_with_k(k, init_state, train_subsets, testset, device)
            all_results[k].append(curve)

    # —— 画均值 —— #
    plt.figure(figsize=(8, 6))
    for k in KS:
        arr = np.array(all_results[k])  # shape: [REPEATS, 6]
        mean = arr.mean(axis=0)         # 直接取均值
        plt.plot(SAMPLE_ROUNDS, mean, marker="o", label=f"k={k}")

    plt.xlabel("Round")
    plt.ylabel("Loss")
    plt.title("SCAFFOLD: Loss vs Round (mean over runs)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
