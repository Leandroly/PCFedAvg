import torch
from torch.utils.data import DataLoader
from copy import deepcopy

class FedVIClient:
    def __init__(self, cid, model, dataset, *, lr, batch_size, device):
        self.cid = cid
        self.device = torch.device(device)
        self.model = deepcopy(model).to(self.device)
        self.dataset = dataset
        self.lr = lr
        self.batch_size = batch_size

        with torch.no_grad():
            self.c_local = {name: torch.zeros_like(p, device=self.device)
                            for name, p in self.model.named_parameters()}
            self.c_global = {name: torch.zeros_like(p, device=self.device)
                             for name, p in self.model.named_parameters()}
    
    def set_global_weights(self, state_dict):
        self.model.load_state_dict({k: v.to(self.device) for k, v in state_dict.items()}, strict=True)
    
    def set_global_control(self, c_global):
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                self.c_global[name].copy_(c_global[name].to(self.device))

    def get_state(self):
        return {k: v.detach().cpu() for k, v in self.model.state_dict().items()}

    def num_samples(self):
        return len(self.dataset)
    
    def _loader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size,
                          shuffle=True, drop_last=False)
    
    def train_one_round(self, *, local_epochs=1):
        self.model.train()
        loss_fn = torch.nn.CrossEntropyLoss()
        opt = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        loader = self._loader()

        with torch.no_grad():
            x = {k: v.clone() for k, v in self.model.state_dict().items()}

        K = 0