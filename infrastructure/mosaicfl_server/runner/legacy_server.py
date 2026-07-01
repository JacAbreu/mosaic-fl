"""legacy_server.py — Servidor Flower legado para desenvolvimento local (python -m infrastructure.mosaicfl_server)."""
import logging
import os
import signal
import threading
from typing import Dict, Optional

import flwr as fl
import torch

from mosaicfl.core.config import FED_CFG, RUNTIME_CFG
from mosaicfl.core.federated import get_evaluate_fn, weighted_average_accuracy, weighted_average_loss
from mosaicfl.core.model import SimplifiedBEHRT

from ..config_loader import get_config_loader
from ..strategy import ProductionFedProxStrategy
from infrastructure.shared.checkpoint_store import get_checkpoint_store

from .checkpoint_io import _load_standard_vocab
from .config import CHECKPOINT_DIR, LOG_DIR, SERVER_ADDRESS, _health
from .health import write_health_status

logger = logging.getLogger(__name__)


class FederatedServer:
    """Servidor Flower de produção para MOSAIC-FL."""

    def __init__(
        self,
        address: str = SERVER_ADDRESS,
        num_rounds: int = FED_CFG.num_rounds,
        min_clients: int = FED_CFG.min_available_clients,
        proximal_mu: float = FED_CFG.proximal_mu,
    ):
        self.address = address
        self.num_rounds = num_rounds
        self.min_clients = min_clients
        self.proximal_mu = proximal_mu
        self.global_model = self._init_model()
        self.test_loader = None
        self._shutdown_event = threading.Event()

    def _init_model(self) -> torch.nn.Module:
        model = SimplifiedBEHRT(use_cls_token=True).to(RUNTIME_CFG.device)
        logger.info(
            "model_initialized",
            extra={"param_count": sum(p.numel() for p in model.parameters()), "device": str(RUNTIME_CFG.device)},
        )
        return model

    def _load_test_data(self) -> Optional[torch.utils.data.DataLoader]:
        """Carrega dados de teste holdout do banco para avaliação global a cada round.

        Usa FL_TEST_HOLDOUT_FRACTION (padrão 0.1) de cada hospital como holdout.
        Requer FL_DB_URL configurado; retorna None silenciosamente caso contrário.

        Nota: em FL real, o servidor não deve ter acesso a dados de pacientes.
        Esta implementação é válida para simulação local (TCC) e para um eventual
        conjunto de validação compartilhado formalmente acordado entre hospitais.
        """
        if not RUNTIME_CFG.db_url:
            logger.info("_load_test_data: FL_DB_URL ausente — avaliação global desativada")
            return None

        try:
            from mosaicfl.core.preprocessor import SequencePipeline
            from mosaicfl.core.config import MAX_SEQ_LEN
            from torch.utils.data import DataLoader, TensorDataset

            holdout_fraction = float(os.getenv("FL_TEST_HOLDOUT_FRACTION", "0.1"))
            batch_size = int(os.getenv("FL_BATCH_SIZE", "32"))

            pipeline = SequencePipeline(
                connection_string=RUNTIME_CFG.db_url,
                max_seq_len=MAX_SEQ_LEN,
            )
            hospital_data = pipeline.build_per_hospital()

            if not hospital_data:
                logger.warning("_load_test_data: SequencePipeline retornou vazio")
                return None

            rng = torch.Generator()
            rng.manual_seed(42)

            test_seqs_list, test_lbls_list = [], []
            for _hospital_id, (seqs, labels, _vocab) in hospital_data.items():
                n = len(seqs)
                if n < 10:
                    continue
                perm = torch.randperm(n, generator=rng)
                n_test = max(1, int(holdout_fraction * n))
                test_seqs_list.append(seqs[perm[:n_test]])
                test_lbls_list.append(labels[perm[:n_test]])

            if not test_seqs_list:
                logger.warning("_load_test_data: nenhuma amostra de teste coletada")
                return None

            test_seqs = torch.cat(test_seqs_list, dim=0)
            test_lbls = torch.cat(test_lbls_list, dim=0)

            loader = DataLoader(
                TensorDataset(test_seqs, test_lbls),
                batch_size=batch_size,
            )
            logger.info(
                "_load_test_data_ready n=%d holdout_fraction=%.2f hospitals=%d",
                len(test_seqs), holdout_fraction, len(test_seqs_list),
            )
            return loader

        except Exception as exc:
            logger.warning("_load_test_data_error %s — avaliação global desativada", exc)
            return None

    def _signal_handler(self, signum, frame):
        logger.info("signal_received", extra={"signum": signum})
        self._shutdown_event.set()

    def _on_round_start(self, round_num: int, runtime_config: dict) -> None:
        """Callback chamado pela strategy antes de cada round."""
        write_health_status("running", round_num=round_num)
        if runtime_config:
            logger.debug("Round %s config: %s", round_num, runtime_config)

    def _on_round_complete(self, round_num: int, metrics: dict) -> None:
        """Callback chamado pela strategy após aggregate_evaluate — publica no HealthServer."""
        _health.set_round_metrics(round_num, metrics)
        logger.debug("round_metrics_published", extra={"round": round_num})

    def start(self):
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        _health.start()

        evaluate_fn = None
        self.test_loader = self._load_test_data()
        if self.test_loader is not None:
            evaluate_fn = get_evaluate_fn(self.test_loader)
            logger.info("evaluate_fn_enabled")

        config_loader = get_config_loader()

        # Tenta recuperar vocab do checkpoint mais recente (se houver)
        recovered_vocab: Dict = {}
        existing_ckpts = sorted(CHECKPOINT_DIR.glob("round_*.pt"))
        if existing_ckpts:
            try:
                ckpt = torch.load(existing_ckpts[-1], map_location="cpu", weights_only=True)
                if isinstance(ckpt, dict) and "model_state" in ckpt:
                    recovered_vocab = ckpt.get("vocab", {})
                    self.global_model.load_state_dict(ckpt["model_state"], strict=False)
                    logger.info(
                        "legacy_checkpoint_restored",
                        extra={"path": str(existing_ckpts[-1]), "vocab_size": len(recovered_vocab)},
                    )
            except Exception as exc:
                logger.warning("legacy_checkpoint_load_error", extra={"error": str(exc)})

        # Fallback: se não há checkpoint ou vocab ausente, carrega standard_vocab.json
        if not recovered_vocab:
            recovered_vocab = _load_standard_vocab()

        strategy = ProductionFedProxStrategy(
            global_model=self.global_model,
            vocab=recovered_vocab,
            config_loader=config_loader,
            checkpoint_store=get_checkpoint_store(RUNTIME_CFG.db_url),
            on_round_start=self._on_round_start,
            on_round_complete=self._on_round_complete,
            test_loader=self.test_loader,
            proximal_mu=self.proximal_mu,
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=self.min_clients,
            min_evaluate_clients=self.min_clients,
            min_available_clients=self.min_clients,
            evaluate_fn=evaluate_fn,
            evaluate_metrics_aggregation_fn=weighted_average_accuracy,
            fit_metrics_aggregation_fn=weighted_average_loss,
            on_fit_config_fn=lambda rnd: {"proximal_mu": self.proximal_mu, "round": rnd},
        )

        logger.info(
            "server_startup",
            extra={
                "address": self.address,
                "rounds": self.num_rounds,
                "min_clients": self.min_clients,
                "proximal_mu": self.proximal_mu,
                "device": str(RUNTIME_CFG.device),
                "checkpoint_dir": str(CHECKPOINT_DIR),
            },
        )

        write_health_status("starting")

        from infrastructure.shared.tls import get_server_certs, tls_enabled
        certs = get_server_certs()
        logger.info(
            "server_tls",
            extra={"enabled": tls_enabled()},
        )

        try:
            fl.server.start_server(
                server_address=self.address,
                config=fl.server.ServerConfig(num_rounds=self.num_rounds),
                strategy=strategy,
                certificates=certs,
            )
        except Exception as e:
            logger.error("server_error", extra={"error": str(e)})
            write_health_status("error", clients=0)
            raise
        finally:
            write_health_status("stopped", round_num=strategy.round_counter)
            logger.info("server_stopped", extra={"rounds_completed": strategy.round_counter})
