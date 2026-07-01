"""factory.py — Cria a fonte de dados correta via variável de ambiente ou parâmetro explícito."""
import logging
import os
from typing import Optional

from .base import DataSource
from .csv_source import CSVDataSource
from .sgbd import SGBDDataSource
from .simulated import SimulatedDataSource

logger = logging.getLogger(__name__)


class DataSourceFactory:
    """Factory para criar a fonte de dados correta via variável de ambiente."""

    _registry = {
        "simulated": SimulatedDataSource,
        "sgbd": SGBDDataSource,
        "csv": CSVDataSource,
    }

    @classmethod
    def create(cls, source_type: Optional[str] = None, **kwargs) -> DataSource:
        """
        Cria uma fonte de dados.

        Args:
            source_type: 'simulated', 'sgbd', 'csv'. Se None, lê FL_DATA_SOURCE.
            **kwargs: Parâmetros específicos da fonte.

        Returns:
            Instância de DataSource pronta para uso.

        Raises:
            ValueError: Se o tipo não for suportado.
        """
        source_type = (source_type or os.getenv("FL_DATA_SOURCE", "simulated")).lower().strip()

        if source_type not in cls._registry:
            raise ValueError(
                f"Fonte de dados '{source_type}' não suportada. "
                f"Opções: {list(cls._registry.keys())}"
            )

        source_class = cls._registry[source_type]
        instance = source_class(**kwargs)

        # Validação automática
        ok, msg = instance.validate()
        if not ok:
            logger.error(f"[FACTORY] Validação falhou: {msg}")
            raise RuntimeError(f"Fonte de dados inválida: {msg}")

        logger.info(f"[FACTORY] Fonte criada: {source_type} — {msg}")
        return instance

    @classmethod
    def register(cls, name: str, source_class: type):
        """Registra uma nova fonte de dados (extensibilidade)."""
        cls._registry[name] = source_class
        logger.info(f"[FACTORY] Fonte registrada: {name}")

    @classmethod
    def available_sources(cls) -> list:
        """Lista fontes disponíveis."""
        return list(cls._registry.keys())
