import torch, random, numpy as np
import matplotlib.pyplot as plt
import copy

from src.utils.config import DATASET, MODEL, TRAINING, OPTIMIZER, LOSS_FN
from src.utils.models import OneNN
from src.client.FedAvg import FedAvgClient
from src.server.FedAvg import FedAvgServer
from generate_data import mnist_subsets


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_with_k(k_value, init_state, train_subsets, testset, device):
    # ---- model（从相同初始权重恢复）----
    global_model = OneNN(in_dim=MODEL["in_dim"], num_classes=MODEL["num_classes"])
    global_model.load_state_dict(copy.deepcopy(init_state))

    clients = [
        FedAvgClient(
            cid=i, model=global_model, dataset=train_subsets[i],
            lr=OPTIMIZER["lr"], batch_size=TRAINING["train_batch_size"], device=device
        )
        for i in range(DATASET["num_clients"])
    ]
    server = FedAvgServer(global_model, clients, device=device)

    # ---- r=0 baseline ----
    metrics = server.evaluate_global(
        dataset=testset, batch_size=TRAINING["eval_batch_size"],
        device=device, loss_fn=LOSS_FN,
    )
    loss_history = [metrics["loss"]]
    print(f"[k={k_value}, Round 0] acc={metrics['accuracy']:.4f} loss={metrics['loss']:.4f}")

    # ---- training loop ----
    for r in range(TRAINING["rounds"]):
        server.run_round(fraction=TRAINING["fraction"], local_epochs=k_value)
        metrics = server.evaluate_global(
            dataset=testset, batch_size=TRAINING["eval_batch_size"],
            device=device, loss_fn=LOSS_FN,
        )
        loss_history.append(metrics["loss"])
        if (r + 1) % 10 == 0 or r == TRAINING["rounds"] - 1:
            print(f"[k={k_value}, Round {r+1}] acc={metrics['accuracy']:.4f} loss={metrics['loss']:.4f}")
    return loss_history

def main():
    set_seed(TRAINING["seed"])
    device = TRAINING["device"] if torch.cuda.is_available() or TRAINING["device"] != "cuda" else "cpu"

    train_subsets, testset = mnist_subsets(
        n_clients=DATASET["num_clients"], scheme=DATASET["partition"],
        alpha=DATASET["dirichlet_alpha"], seed=TRAINING["seed"], root=DATASET["root"]
    )

    # —— 只初始化一次模型，并保存初始权重 ——
    base_model = OneNN(in_dim=MODEL["in_dim"], num_classes=MODEL["num_classes"])
    init_state = copy.deepcopy(base_model.state_dict())

    ks = [1, 5, 10]
    results = {k: run_with_k(k, init_state, train_subsets, testset, device) for k in ks}

    # —— 画图（含 r=0；每 10 轮取点）——
    plt.figure(figsize=(8, 6))
    for k, losses in results.items():
        idxs = list(range(0, len(losses), 10))
        if (len(losses) - 1) not in idxs: idxs.append(len(losses) - 1)
        xs = idxs                       # 这里 xs 已经是 round 编号（含 0）
        ys = [losses[i] for i in idxs]
        plt.plot(xs, ys, marker="o", label=f"k={k}")
    plt.xlabel("Round"); plt.ylabel("Loss"); plt.title("Loss vs Round for different k (same init)")
    plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()

if __name__ == "__main__":
    main()