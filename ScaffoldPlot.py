# main_scaffold_vi_personal.py
import torch, random, numpy as np
import matplotlib.pyplot as plt
import copy

from src.utils.config import DATASET, MODEL, TRAINING, OPTIMIZER, LOSS_FN
from src.utils.models import OneNN
from src.client.ScaffoldVI import ScaffoldVIClient
from src.server.ScaffoldVI import ScaffoldVIServer
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


def eta_schedule(r: int) -> float:
    return 0.2


def _zeros_like_state(state_dict: dict):
    return {k: torch.zeros_like(v, device="cpu") for k, v in state_dict.items()}


def print_block(state_dict: dict, client_id: int, max_show: int = 5, ci_norm: float | None = None):
    header = f"---- Client[{client_id}] block snapshot"
    if ci_norm is not None:
        header += f" | ||c_i||={ci_norm:.4f}"
    print(header + " ----")
    for name, t in state_dict.items():
        flat = t.view(-1).float()
        head = ", ".join(f"{v:.4f}" for v in flat[:max_show].tolist())
        tail = " ..." if flat.numel() > max_show else ""
        print(f"{name:20s} shape={list(t.shape)!r}  values=[{head}{tail}]")
    print("---------------------------------------------------")


def evaluate_personalized_common_test(server, clients, testset, *, batch_size, device, loss_fn):
    results = []
    for i, c in enumerate(clients):
        block_i = server.get_block(i)  # CPU tensors
        c.set_broadcast(block_i, _zeros_like_state(block_i))
        res = c.evaluate(testset, batch_size=batch_size, device=device, loss_fn=loss_fn)
        results.append(res)

    loss = float(np.mean([r["loss"] for r in results]))
    acc  = float(np.mean([r["accuracy"] for r in results]))
    tot  = int(np.sum([r["num_samples"] for r in results]))
    return {"loss": loss, "accuracy": acc, "num_samples": tot}, results

def lambda_schedule(r: int) -> float:
    return 0.1


def run_scaffoldvi_personal_with_k(k_value, init_state, train_subsets, testset, device):
    base_model = OneNN(in_dim=MODEL["in_dim"], num_classes=MODEL["num_classes"])
    base_model.load_state_dict(copy.deepcopy(init_state))

    clients = [
        ScaffoldVIClient(
            cid=i,
            model=base_model,
            dataset=train_subsets[i],
            lr=OPTIMIZER["lr"],
            batch_size=TRAINING["train_batch_size"],
            device=device,
        )
        for i in range(DATASET["num_clients"])
    ]
    server = ScaffoldVIServer(base_model, clients, device=device, gamma_g=1.0)

    metrics, _ = evaluate_personalized_common_test(
        server, clients, testset,
        batch_size=TRAINING["eval_batch_size"], device=device, loss_fn=LOSS_FN
    )
    losses = {0: metrics["loss"]}
    print(f"[ScaffoldVI-Personal k={k_value}, Round 0] acc={metrics['accuracy']:.4f} loss={metrics['loss']:.4f}")

    for r in range(1, TRAINING["rounds"] + 1):
        eta_r = eta_schedule(r)
        lam_r = lambda_schedule(r)
        stats = server.run_round(
            fraction=TRAINING["fraction"],
            local_epochs=k_value,
            eta_r=eta_r,
            lambda_reg=lam_r,
            round_idx=r,
        )
        reg_val = server.global_reg_value(lam_r)
        print(f"[k={k_value}, Round {r}] selected={stats['selected']} | "
                f"samples={stats['total_samples']} | eta={eta_r:.3f} | lambda={lam_r:.3f} | "
                f"RegSum={reg_val:.3e}")

        if r % 10 == 0:
            for idx in range(DATASET["num_clients"]):
                block = server.get_block(idx)
                ci = server.get_ci(idx)
                ci_norm = float(sum(v.detach().float().norm().item() for v in ci.values()))
                print_block(block, client_id=idx, max_show=5, ci_norm=ci_norm)

        metrics, _ = evaluate_personalized_common_test(
            server, clients, testset,
            batch_size=TRAINING["eval_batch_size"], device=device, loss_fn=LOSS_FN
        )
        if r in SAMPLE_ROUNDS:
            losses[r] = metrics["loss"]

        if (r % 10 == 0) or (r == TRAINING["rounds"]):
            print(f"[ScaffoldVI-Personal k={k_value}, Round {r}] "
                  f"acc={metrics['accuracy']:.4f} loss={metrics['loss']:.4f}")

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
            curve = run_scaffoldvi_personal_with_k(k, init_state, train_subsets, testset, device)
            all_results[k].append(curve)

    plt.figure(figsize=(8, 6))
    for k in KS:
        arr = np.array(all_results[k])
        mean = arr.mean(axis=0)
        plt.plot(SAMPLE_ROUNDS, mean, marker="o", label=f"k={k}")

    plt.xlabel("Round")
    plt.ylabel("Personalized Loss (avg over clients)")
    plt.title("ScaffoldVI-Personal (m blocks, no cross-client averaging)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
