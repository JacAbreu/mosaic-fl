"""
datasource — Strategy Pattern para fontes de dados do cliente federado.

Suporta:
  - simulated: dados sintéticos para TCC/prototipagem
  - sgbd: conexão PostgreSQL/MySQL/SQLite do hospital
  - csv: arquivo CSV local (fallback para hospitais sem SGBD)
  - fhir: integração HL7 FHIR (futuro)

Uso:
    from infrastructure.mosaicfl_client.datasource import DataSourceFactory

    source = DataSourceFactory.create("sgbd", connection_string="postgresql://...")
    loader = source.load()

Submódulos:
  base.py        — DataSource (interface ABC) + constantes padrão
  simulated.py     — SimulatedDataSource
  sgbd.py            — SGBDDataSource (produção, via SequencePipeline)
  csv_source.py         — CSVDataSource
  factory.py               — DataSourceFactory
"""
from .base import DataSource
from .csv_source import CSVDataSource
from .factory import DataSourceFactory
from .sgbd import SGBDDataSource
from .simulated import SimulatedDataSource

__all__ = [
    "DataSource",
    "SimulatedDataSource",
    "SGBDDataSource",
    "CSVDataSource",
    "DataSourceFactory",
]
