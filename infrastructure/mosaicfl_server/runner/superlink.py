"""superlink.py — ServerApp para o modo de produção via flower-superlink.

Entry point para: flwr run . <federation>
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import flwr as fl
import torch
from flwr.common import Context
from flwr.server import ServerApp, ServerAppComponents, ServerConfig

from mosaicfl.core.config import FED_CFG, RUNTIME_CFG
from mosaicfl.core.federated import weighted_average_evaluate_metrics, weighted_average_loss
from mosaicfl.core.model import SimplifiedBEHRT

from ..config_loader import get_config_loader
from ..state_store import TrainingStateStore
from ..strategy import ProductionFedProxStrategy
from infrastructure.shared.checkpoint_store import get_checkpoint_store

from .checkpoint_io import _load_standard_vocab
from .config import LOG_DIR, _health
from .health import write_health_status

logger = logging.getLogger(__name__)

# ServerApp roda como subprocesso próprio (flower-superexec --plugin-type serverapp),
# cuja stdout/stderr não aparece nem no terminal do "flwr run", nem no do
# "flower-superlink" — gap de observabilidade real (achado em 2026-07-05/06,
# nenhum log da estratégia/aggregate_fit/aggregate_evaluate era visível em
# lugar nenhum). Log em arquivo próprio, capturando o logger raiz (pega tudo:
# core.py, watchdog_mixin.py, calibration_mixin.py, este módulo).
#
# Caminho relativo ao CWD do processo (não a __file__): este módulo roda a
# partir do FAB extraído (~/.flwr/apps/...), não do checkout do projeto —
# Path(__file__).parent... apontaria pra dentro do FAB, não pro projeto real.
# LOG_DIR/CHECKPOINT_DIR (config.py) já usam esse mesmo padrão relativo ao CWD
# com sucesso (round_N_metrics.json aparece certo em logs/ do projeto).
_EXPERIMENTS_LOG_DIR = Path("experiments/logs")
_EXPERIMENTS_LOG_DIR.mkdir(parents=True, exist_ok=True)
_serverapp_log_file = _EXPERIMENTS_LOG_DIR / f"serverapp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
_root_logger = logging.getLogger()
if not any(isinstance(h, logging.FileHandler) and getattr(h, "_mosaicfl_serverapp", False)
           for h in _root_logger.handlers):
    _file_handler = logging.FileHandler(_serverapp_log_file, encoding="utf-8")
    _file_handler._mosaicfl_serverapp = True
    _file_handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s | %(message)s"
    ))
    _root_logger.addHandler(_file_handler)
    _root_logger.setLevel(logging.INFO)
    logger.info("serverapp_log_file_iniciado path=%s", _serverapp_log_file)


def _make_server_components(context: Context) -> ServerAppComponents:
    """
    Factory chamada pelo SuperLink a cada execução de ServerApp.

    Lê run_config (pyproject.toml / --run-config), recupera estado da sessão
    anterior (se houver) e reconstrói a estratégia FedProx com tracker restaurado
    e pesos do último checkpoint carregados como initial_parameters.

    TLS é responsabilidade do flower-superlink — não é configurado aqui.
    """
    num_rounds = int(context.run_config.get("num-rounds", FED_CFG.num_rounds))
    min_clients = int(context.run_config.get("min-clients", FED_CFG.min_available_clients))
    proximal_mu = float(context.run_config.get("proximal-mu", FED_CFG.proximal_mu))
    local_epochs = int(context.run_config.get("local-epochs", FED_CFG.local_epochs))
    round_timeout = int(context.run_config.get("round-timeout-seconds", 300))

    # "ajuste" (default) ou "treinamento_real" — mesma semântica do FL_RUN_CLASSIFICATION
    # do Caminho A (manual_loop.py). Sem isso, um dump do banco não distingue
    # tuning/debug de resultado citável na defesa.
    run_classification = str(context.run_config.get("run-classification", "ajuste")).strip().lower()
    if run_classification not in ("ajuste", "treinamento_real"):
        logger.warning(
            "run_classification_invalido valor=%r (esperado 'ajuste' ou 'treinamento_real') "
            "— usando 'ajuste' para não classificar erroneamente um run como resultado formal.",
            run_classification,
        )
        run_classification = "ajuste"

    # ── Recovery de estado ───────────────────────────────────────────────────
    state_path = LOG_DIR / "training_state.json"
    state_store = TrainingStateStore(state_path)
    previous_state = state_store.load()

    # Marca nova sessão como "running" imediatamente — se crashar, próximo load detecta
    previous_state.status = "running"
    state_store.save(previous_state)

    # ── Modelo: carrega checkpoint da sessão anterior se disponível ──────────
    model = SimplifiedBEHRT(use_cls_token=True).to(RUNTIME_CFG.device)
    initial_parameters: Optional[fl.common.Parameters] = None
    recovered_vocab: Dict = {}

    if previous_state.last_checkpoint:
        ckpt_path = Path(previous_state.last_checkpoint)
        if ckpt_path.exists():
            try:
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
                # Novo formato: {"model_state": ..., "vocab": ...}
                # Legado: state_dict puro (sem chave "model_state")
                if isinstance(ckpt, dict) and "model_state" in ckpt:
                    state_dict      = ckpt["model_state"]
                    recovered_vocab = ckpt.get("vocab", {})
                else:
                    state_dict = ckpt
                    logger.warning(
                        "checkpoint_legacy_format — vocab ausente; "
                        "tentando carregar standard_vocab.json como fallback"
                    )
                model.load_state_dict(state_dict, strict=False)
                initial_parameters = fl.common.ndarrays_to_parameters(
                    [v.cpu().detach().numpy().copy() for v in state_dict.values()]
                )
                logger.info(
                    "checkpoint_loaded_for_recovery",
                    extra={
                        "checkpoint": str(ckpt_path),
                        "last_round": previous_state.last_round,
                        "vocab_size": len(recovered_vocab),
                    },
                )
            except Exception as exc:
                logger.warning("checkpoint_load_error", extra={"error": str(exc)})

    # Fallback: se o checkpoint não trouxe vocab (primeiro round ou legado), carrega standard_vocab
    if not recovered_vocab:
        recovered_vocab = _load_standard_vocab()

    # Sem vocab nenhum, o servidor enviaria vocab_json vazio pra todos os clientes — cada um
    # cairia de volta a construir seu próprio vocab local (mesmo problema que motivou distribuir
    # o vocab pelo protocolo FL). Falha aqui, no servidor, é mais cedo e mais claro que deixar
    # o erro aparecer disperso em cada cliente.
    if not recovered_vocab:
        raise RuntimeError(
            "Nenhum vocabulário padrão disponível (nem checkpoint anterior, nem "
            "checkpoints/standard_vocab.json). Execute scripts/build_standard_vocab.py "
            "antes de iniciar o ServerApp."
        )

    config_loader = get_config_loader()
    checkpoint_store = get_checkpoint_store(RUNTIME_CFG.db_url)
    _health.start()

    training_id = checkpoint_store.register_training(
        algorithm="FedProx",
        log_file="",
        n_rounds_max=num_rounds,
        checkpoint_criterion=FED_CFG.checkpoint_criterion,
        partition_mode="natural",
        run_classification=run_classification,
    )
    logger.info(
        "training_registered",
        extra={
            "training_id": training_id,
            "algorithm": "FedProx",
            "n_rounds_max": num_rounds,
            "run_classification": run_classification,
            "run_id": context.run_id,
        },
    )

    strategy = ProductionFedProxStrategy(
        global_model=model,
        vocab=recovered_vocab,
        config_loader=config_loader,
        state_store=state_store,
        checkpoint_store=checkpoint_store,
        round_timeout=round_timeout,
        training_id=training_id,
        num_rounds=num_rounds,
        run_id=context.run_id,
        on_round_start=lambda rnd, cfg: write_health_status("running", round_num=rnd),
        on_round_complete=lambda rnd, metrics: _health.set_round_metrics(rnd, metrics),
        proximal_mu=proximal_mu,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=min_clients,
        min_evaluate_clients=min_clients,
        min_available_clients=min_clients,
        initial_parameters=initial_parameters,
        evaluate_metrics_aggregation_fn=weighted_average_evaluate_metrics,
        fit_metrics_aggregation_fn=weighted_average_loss,
        # vocab_json: distribui o vocabulário canônico a cada rodada, direto pelo protocolo FL —
        # sem isso, cada cliente construía seu próprio vocab local (tamanhos incompatíveis entre
        # hospitais), causando falha silenciosa na agregação. Ver FedProxClient._ensure_data().
        on_fit_config_fn=lambda rnd: {
            "proximal_mu": proximal_mu,
            "local_epochs": local_epochs,
            "round": rnd,
            "vocab_json": json.dumps(recovered_vocab),
        },
        # extract_rag_patterns: pede ao cliente pra extrair perfis prototípicos (RAG) só
        # na última rodada configurada — caro pra repetir a cada round (forward com atenção
        # sobre o val_loader inteiro, uma vez por classe). Se a convergência disparar antes
        # da última rodada, os padrões não são coletados nessa run — limitação conhecida,
        # aceitável dado que a maioria dos runs reais roda até o limite de rodadas mesmo
        # (ver docs/Linha_do_Tempo_MOSAIC-FL.md, treinamentos convergem em média aos 40-46
        # rounds do Caminho A, bem acima do budget típico de testes do Caminho B).
        on_evaluate_config_fn=lambda rnd: {
            "round": rnd,
            "vocab_json": json.dumps(recovered_vocab),
            "extract_rag_patterns": rnd >= num_rounds,
        },
    )

    write_health_status("starting")
    logger.info(
        "server_startup_superlink",
        extra={
            "rounds": num_rounds,
            "min_clients": min_clients,
            "proximal_mu": proximal_mu,
            "round_timeout": round_timeout,
            "previous_status": previous_state.status,
            "recovered_from_round": previous_state.last_round,
        },
    )
    return ServerAppComponents(
        strategy=strategy,
        config=ServerConfig(num_rounds=num_rounds),
    )


# Entry point para: flwr run . <federation>
app = ServerApp(server_fn=_make_server_components)
