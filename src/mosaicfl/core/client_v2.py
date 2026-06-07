"""
Cliente Flower com FedProx para treinamento local federado (FedProxClient).

Recebe pesos globais do servidor, treina localmente com dados EHR do hospital,
e devolve apenas os pesos atualizados — nunca os dados brutos.
Usa state_dict() para sincronizar todos os tensores (treinaveis + buffers).
"""
import logging
import numpy as np
import torch
import flwr as fl
from torch.utils.data import DataLoader, TensorDataset
from collections import OrderedDict
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

from mosaicfl.core.model_v2 import SimplifiedBEHRT
from .config import FED_CFG, RUNTIME_CFG


class FedProxClient(fl.client.NumPyClient):
    def __init__(self, client_id: int, train_loader: DataLoader, val_loader: DataLoader):
        self.client_id = client_id
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = SimplifiedBEHRT(use_cls_token=True).to(RUNTIME_CFG.device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=FED_CFG.lr)
        self.global_params = None  # para termo proximal

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """
        Carrega pesos globais no modelo local.
        Usa model.parameters() para garantir que apenas parâmetros treináveis
        sejam sincronizados (não buffers como running_mean de BN).
        """
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        # strict=False permite carregar apenas os parâmetros fornecidos,
        # ignorando buffers que não estão na lista de parâmetros treináveis.
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning("params_missing", extra={"client_id": self.client_id, "keys": missing})
        if unexpected:
            logger.warning("params_unexpected", extra={"client_id": self.client_id, "keys": unexpected})

        # Armazena cópia dos parâmetros globais para o termo proximal
        self.global_params = [p.clone().detach().to(RUNTIME_CFG.device) for p in self.model.parameters()]

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """
        Retorna apenas os parâmetros treináveis (não buffers).
        Reduz tráfego de rede e evita inconsistências no servidor.
        """
        #return [p.detach().cpu().numpy() for p in self.model.parameters()]
        return [v.cpu().detach().numpy().copy() for v in self.model.state_dict().values()]

    def _proximal_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Adiciona termo proximal do FedProx."""
        if self.global_params is None:
            return loss
        proximal_term = 0.0
        for local_w, global_w in zip(self.model.parameters(), self.global_params):
            proximal_term += torch.norm(local_w - global_w, p=2) ** 2
        return loss + (FED_CFG.proximal_mu / 2) * proximal_term

    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        self.set_parameters(parameters)
        self.model.train()
        epoch_losses = []

        for epoch in range(FED_CFG.local_epochs):
            running_loss = 0.0
            total_samples = 0
            for batch_x, batch_y in self.train_loader:
                try:
                    batch_x, batch_y = batch_x.to(RUNTIME_CFG.device), batch_y.to(RUNTIME_CFG.device)
                    self.optimizer.zero_grad()
                    outputs = self.model(batch_x)
                    loss = self.criterion(outputs, batch_y)
                    loss = self._proximal_loss(loss)
                    loss.backward()
                    self.optimizer.step()
                    # Normaliza pelo número real de amostras no batch
                    running_loss += loss.item() * batch_y.size(0)
                    total_samples += batch_y.size(0)
                except Exception as e:
                    print(f"[Cliente {self.client_id}] Erro no batch: {e}")
                    continue  # pula batch problemático, não quebra o cliente

            epoch_loss = running_loss / total_samples if total_samples > 0 else 0.0
            epoch_losses.append(epoch_loss)

        return self.get_parameters(config), len(self.train_loader.dataset), {"loss": sum(epoch_losses)/len(epoch_losses)}

    def evaluate(self, parameters: List[np.ndarray], config: Dict) -> Tuple[float, int, Dict]:
        self.set_parameters(parameters)
        self.model.eval()
        correct, total, loss_sum = 0, 0, 0.0
        with torch.no_grad():
            for batch_x, batch_y in self.val_loader:
                batch_x, batch_y = batch_x.to(RUNTIME_CFG.device), batch_y.to(RUNTIME_CFG.device)
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                loss_sum += loss.item() * batch_y.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

        accuracy = correct / total if total > 0 else 0
        avg_loss = loss_sum / total if total > 0 else 0.0
        return float(avg_loss), total, {"accuracy": accuracy, "client_id": self.client_id}


def create_client_fn(client_id: int, train_data: torch.Tensor, train_labels: torch.Tensor,
                     val_data: torch.Tensor, val_labels: torch.Tensor):
    """Factory para criar clientes com seus respectivos DataLoaders."""
    train_dataset = TensorDataset(train_data, train_labels)
    val_dataset = TensorDataset(val_data, val_labels)
    train_loader = DataLoader(train_dataset, batch_size=FED_CFG.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=FED_CFG.batch_size)
    return FedProxClient(client_id, train_loader, val_loader)