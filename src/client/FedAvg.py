import torch
from torch.utils.data import DataLoader
from copy import deepcopy

class FedAvgClient:
    def __init__(self, cid, model, dataset, *, lr, batch_size, device):
        self.cid = cid
        self.device = torch.device(device)
        self.model = deepcopy(model).to(self.device)
        self.dataset = dataset
        self.batch_size = batch_size
        self.lr = lr

    def set_global_weights(self, state_dict):
        self.model.load_state_dict({k:v.to(self.device) for k,v in state_dict.items()}, strict=True)

    def get_state(self):
        return {k: v.detach().cpu() for k, v in self.model.state_dict().items()}

    def num_samples(self):
        return len(self.dataset)

    def _loader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)

    def train_one_round(self, *, local_epochs=1):
        self.model.train()
        opt = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        loss_fn = torch.nn.CrossEntropyLoss()
        loader = self._loader()

        for _ in range(local_epochs):
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                opt.zero_grad()
                logits = self.model(x)
                loss = loss_fn(logits, y)
                loss.backward()
                opt.step()

        return {
            "cid": self.cid,
            "num_samples": self.num_samples(),
            "state": self.get_state(),
        }
    
    @torch.no_grad()
    def evaluate(self, dataset, *, batch_size, device, loss_fn):
        self.model.eval()
        device = torch.device(self.device)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        total_loss, correct, total = 0.0, 0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = self.model(x)
            loss = loss_fn(logits, y)
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        return {
            "loss": total_loss / len(loader),
            "accuracy": correct / total,
            "num_samples": total
        }
