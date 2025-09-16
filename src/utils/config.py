# config.py
from torch import optim, nn

DATASET = {
    "name": "mnist",
    "root": "./data",
    "partition": "noniid",
    "dirichlet_alpha": 0.1,
    "num_clients": 20,
}

MODEL = {
    "name": "OneNN",
    "in_dim": 28*28,
    "num_classes": 10,
}

TRAINING = {
    "seed": 0,
    "device": "cuda",
    "rounds": 50,
    "fraction": 0.2,
    "local_epochs": 2,
    "train_batch_size": 64,
    "eval_batch_size": 256,
}

OPTIMIZER = {
    "name": "sgd",
    "lr": 0.05,
    "weight_decay": 0.0,
}

LOSS_FN = nn.CrossEntropyLoss()

OPTIMIZERS = {
    "sgd": optim.SGD,
    "adam": optim.Adam,
    "adamw": optim.AdamW,
}

def get_optimizer_fn():
    opt_class = OPTIMIZERS[OPTIMIZER["name"]]
    def _fn(model):
        return opt_class(
            model.parameters(),
            lr=OPTIMIZER["lr"],
            momentum=OPTIMIZER.get("momentum", 0.0),
            weight_decay=OPTIMIZER.get("weight_decay", 0.0)
        )
    return _fn
