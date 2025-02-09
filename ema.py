import torch
import torch.nn as nn
from copy import deepcopy

class EMAModel(nn.Module):
    def __init__(self, model: nn.Module, beta: float = 0.999, device="cuda"):
        super(EMAModel, self).__init__()
        self.ema_model = deepcopy(model)
        self.ema_model.to(device)
        self.ema_model.requires_grad_(False)
        self.beta = beta
    def update(self, model: nn.Module) -> None:
        for ema_param, raw_param in zip(self.ema_model.parameters(), model.parameters()):
            raw_param.to(ema_param.device)
            ema_param.copy_(self.beta * ema_param + (1 - self.beta) * raw_param)
    def ema_inference(self, data) -> torch.Tensor:
        self.ema_model.eval()
        with torch.no_grad():
            return self.ema_model(data.to(self.ema_model.device))