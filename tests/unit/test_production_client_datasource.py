"""
test_production_client_datasource.py

Garante que ProductionClient._load_data_loaders() NÃO faz fallback silencioso
para dados simulados quando a fonte configurada falha.

Comportamento esperado ao falhar:
  - Propaga a exceção → run() trata, seta health="error", aguarda e retenta
  - O round FL NÃO acontece com dados sintéticos sem notificação
"""
import sys
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def _make_client(source_type: str = "sgbd"):
    from infrastructure.mosaicfl_client.runner import ProductionClient
    return ProductionClient(
        server_address="localhost:8080",
        client_id="hospital_test",
        data_source=source_type,
    )


class TestProductionClientDatasource:
    """
    ProductionClient._load_data_loaders() deve falhar explicitamente quando
    a fonte de dados configurada não está disponível.
    """

    def test_sgbd_failure_raises_and_does_not_fall_back(self):
        """Falha no SGBD propaga exceção — sem substituição por dados simulados."""
        client = _make_client("sgbd")

        with patch(
            "infrastructure.mosaicfl_client.runner.DataSourceFactory.create",
            side_effect=RuntimeError("Erro de conexão: host não encontrado"),
        ) as mock_create:
            with pytest.raises(RuntimeError, match="Erro de conexão"):
                client._load_data_loaders()

        # Deve ter tentado criar apenas uma fonte — nunca "simulated" como fallback
        assert mock_create.call_count == 1
        called_with = mock_create.call_args[0][0]
        assert called_with == "sgbd"

    def test_csv_failure_raises_and_does_not_fall_back(self):
        """Falha no CSV propaga exceção — sem substituição por dados simulados."""
        client = _make_client("csv")

        with patch(
            "infrastructure.mosaicfl_client.runner.DataSourceFactory.create",
            side_effect=RuntimeError("Arquivo não encontrado: data/hospital.csv"),
        ) as mock_create:
            with pytest.raises(RuntimeError, match="Arquivo não encontrado"):
                client._load_data_loaders()

        assert mock_create.call_count == 1
        assert mock_create.call_args[0][0] == "csv"

    def test_simulated_source_succeeds_without_fallback(self):
        """Fonte simulada explícita funciona normalmente."""
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        dummy_loader = DataLoader(
            TensorDataset(torch.randint(0, 100, (8, 16)), torch.randint(0, 2, (8,))),
            batch_size=4,
        )
        mock_source = MagicMock()
        mock_source.get_metadata.return_value = {"type": "simulated"}
        mock_source.load.return_value = dummy_loader

        client = _make_client("simulated")

        with patch(
            "infrastructure.mosaicfl_client.runner.DataSourceFactory.create",
            return_value=mock_source,
        ):
            train_loader, val_loader = client._load_data_loaders()

        assert isinstance(train_loader, DataLoader)
        assert isinstance(val_loader, DataLoader)
