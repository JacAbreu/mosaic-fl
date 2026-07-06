"""supernode.py — ClientApp para o modo de produção via flower-supernode.

Entry point para: flower-supernode --superlink <addr> ... (executa flwr-clientapp internamente)
"""
import json
import logging
import os

import flwr as fl
from flwr.client import ClientApp
from flwr.common import Context

from mosaicfl.core.client import FedProxClient

from ..datasource import DataSourceFactory
from .data_utils import _loader_cache, _split_loader, parse_client_id

logger = logging.getLogger(__name__)

# Fontes que dependem de um vocabulário canônico compartilhado entre clientes —
# carregamento adiado pro 1º fit()/evaluate() (vocab só chega via config da rodada).
# Fontes fora desta lista (simulated, csv) continuam com carregamento imediato, como antes.
_VOCAB_DEPENDENT_SOURCES = {"sgbd"}


def _client_fn(context: Context) -> fl.client.Client:
    """
    Factory chamada pelo SuperNode a cada round.

    Lê node_config (--node-config do flower-supernode) para identificar
    o hospital e a fonte de dados. TLS é responsabilidade do SuperNode — não
    é configurado aqui.

    DataLoaders são cacheados entre rounds: a query ao banco (que pode levar minutos)
    ocorre apenas no primeiro round. Dados não mudam durante uma sessão de treinamento.
    """
    client_id_str    = str(context.node_config.get("client-id", str(context.node_id)))
    data_source_type = str(
        context.node_config.get("data-source", os.getenv("FL_DATA_SOURCE", "simulated"))
    )
    hospital_id   = str(context.node_config.get("hospital-id", os.getenv("FL_HOSPITAL_ID", client_id_str)))
    client_id_int = parse_client_id(client_id_str)

    cache_key = (data_source_type, hospital_id)

    if data_source_type in _VOCAB_DEPENDENT_SOURCES:
        source = DataSourceFactory.create(data_source_type, hospital_id=hospital_id)

        def _loader_factory(vocab_json: str):
            if cache_key not in _loader_cache:
                logger.info(
                    "data_loading_start",
                    extra={"client_id": client_id_str, "hospital_id": hospital_id, "data_source": data_source_type},
                )
                vocab = json.loads(vocab_json)
                _loader_cache[cache_key] = _split_loader(source.load(vocab=vocab))
                logger.info(
                    "data_loaded_and_cached",
                    extra={"client_id": client_id_str, "hospital_id": hospital_id, "data_source": data_source_type},
                )
            else:
                logger.debug("data_cache_hit source=%s hospital_id=%s", data_source_type, hospital_id)
            return _loader_cache[cache_key]

        return FedProxClient(client_id=client_id_int, loader_factory=_loader_factory).to_client()

    if cache_key not in _loader_cache:
        logger.info(
            "data_loading_start",
            extra={"client_id": client_id_str, "hospital_id": hospital_id, "data_source": data_source_type},
        )
        source = DataSourceFactory.create(data_source_type, hospital_id=hospital_id)
        _loader_cache[cache_key] = _split_loader(source.load())
        logger.info(
            "data_loaded_and_cached",
            extra={"client_id": client_id_str, "hospital_id": hospital_id, "data_source": data_source_type},
        )
    else:
        logger.debug("data_cache_hit source=%s hospital_id=%s", data_source_type, hospital_id)

    train_loader, val_loader = _loader_cache[cache_key]

    return FedProxClient(
        client_id=client_id_int,
        train_loader=train_loader,
        val_loader=val_loader,
    ).to_client()


# Entry point para: flower-supernode ... (SuperNode executa flwr-clientapp internamente)
app = ClientApp(client_fn=_client_fn)
