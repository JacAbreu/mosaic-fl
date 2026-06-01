"""
Cliente Flower com FedProx para treinamento local em cada hospital.
"""
import numpy as np
import torch
import flwr as fl
from torch.utils.data import DataLoader, TensorDataset
from collections import OrderedDict
from typing import Dict, List, Tuple

from model import SimplifiedBEHRT
from config import *


class FedProxClient(fl.client.NumPyClient):
    def __init__(self, client_id: int, train_loader: DataLoader, val_loader: DataLoader):
        self.client_id = client_id
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = SimplifiedBEHRT().to(DEVICE)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LR)
        self.global_params = None  # para termo proximal

    def set_parameters(self, parameters: List[np.ndarray]):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
        self.global_params = [p.clone().detach() for p in self.model.parameters()]

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def _proximal_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Adiciona termo proximal do FedProx."""
        if self.global_params is None:
            return loss
        proximal_term = 0.0
        for local_w, global_w in zip(self.model.parameters(), self.global_params):
            proximal_term += torch.norm(local_w - global_w, p=2) ** 2
        return loss + (PROXIMAL_MU / 2) * proximal_term

    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        self.set_parameters(parameters)
        self.model.train()
        epoch_losses = []

        for epoch in range(LOCAL_EPOCHS):
            running_loss = 0.0
            for batch_x, batch_y in self.train_loader:
                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                self.optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                loss = self._proximal_loss(loss)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            epoch_losses.append(running_loss / len(self.train_loader))

        return self.get_parameters(config), len(self.train_loader.dataset), {"loss": sum(epoch_losses)/len(epoch_losses)}

    def evaluate(self, parameters: List[np.ndarray], config: Dict) -> Tuple[float, int, Dict]:
        self.set_parameters(parameters)
        self.model.eval()
        correct, total, loss_sum = 0, 0, 0.0
        with torch.no_grad():
            for batch_x, batch_y in self.val_loader:
                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                loss_sum += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

        accuracy = correct / total if total > 0 else 0
        return float(loss_sum / len(self.val_loader)), total, {"accuracy": accuracy, "client_id": self.client_id}


def create_client_fn(client_id: int, train_data: torch.Tensor, train_labels: torch.Tensor,
                     val_data: torch.Tensor, val_labels: torch.Tensor):
    """Factory para criar clientes com seus respectivos DataLoaders."""
    train_dataset = TensorDataset(train_data, train_labels)
    val_dataset = TensorDataset(val_data, val_labels)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    return FedProxClient(client_id, train_loader, val_loader)
