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
from flwr.server import ServerApp, ServerAppComponents, ServerConfig, SimpleClientManager

from mosaicfl.core.config import FED_CFG, MODEL_CFG, RUNTIME_CFG
from mosaicfl.core.federated import weighted_average_evaluate_metrics, weighted_average_loss
from mosaicfl.core.model import SimplifiedBEHRT

from ..config_loader import get_config_loader
from ..state_store import TrainingStateStore
from ..strategy import ProductionFedProxStrategy
from infrastructure.shared.checkpoint_store import get_checkpoint_store
from scripts.build_standard_vocab import build_standard_vocab

from .config import LOG_DIR, _health
from .early_stop_server import EarlyStoppingServer
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
# com sucesso (checkpoints/round_N.pt aparece certo no projeto, não no FAB).
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
    # demo_dim (MODEL_CFG.demo_dim, achado 2026-07-28): DEVE ser idêntico nas duas
    # máquinas (servidor e todos os clientes) — dimensões diferentes quebrariam a
    # agregação de pesos. Default 0 preserva o comportamento histórico.
    model = SimplifiedBEHRT(use_cls_token=True, demo_dim=MODEL_CFG.demo_dim).to(RUNTIME_CFG.device)
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
                        "tentando reconstruir do banco como fallback"
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

    # Fallback: se o checkpoint não trouxe vocab (primeiro round ou legado), reconstrói
    # direto de knowledge.term_dictionary/analyte_references — não de um arquivo local.
    # Um arquivo (checkpoints/standard_vocab.json) ficaria desatualizado silenciosamente
    # toda vez que o vocabulário federado bidirecional inserisse algo novo entre um treino
    # e o próximo (ver strategy/core.py::_discover_and_curate_vocab) — exatamente o tipo de
    # "alguém esquece de regenerar" que motivou o desenho automático em primeiro lugar
    # (achado 2026-07-26, mesma sessão). Banco é fonte única de verdade também no boot.
    if not recovered_vocab:
        try:
            recovered_vocab = build_standard_vocab(RUNTIME_CFG.db_url)
        except Exception as exc:
            logger.warning("vocab_boot_rebuild_error", extra={"error": str(exc)})

    # Sem vocab nenhum, o servidor enviaria vocab_json vazio pra todos os clientes — cada um
    # cairia de volta a construir seu próprio vocab local (mesmo problema que motivou distribuir
    # o vocab pelo protocolo FL). Falha aqui, no servidor, é mais cedo e mais claro que deixar
    # o erro aparecer disperso em cada cliente.
    if not recovered_vocab:
        raise RuntimeError(
            "Nenhum vocabulário padrão disponível (nem checkpoint anterior, nem "
            "reconstrução a partir de knowledge.term_dictionary/analyte_references). "
            "Confira se há analitos ativos no catálogo (knowledge.term_dictionary)."
        )

    config_loader = get_config_loader()
    checkpoint_store = get_checkpoint_store(RUNTIME_CFG.db_url)
    _health.start()

    # local-only-hospital (run-config, default ""): "BPSP"/"HSL" quando este treino
    # roda com um único hospital conectado (min-clients=1) — baseline local pra
    # comparar contra o federado na mesma rede real (migration 030). None = treino
    # federado normal. Declarado explicitamente por quem sobe o SuperLink, não
    # inferido automaticamente — evita marcar errado um treino federado de verdade.
    local_only_hospital = str(context.run_config.get("local-only-hospital", "")).strip().upper() or None
    if local_only_hospital and min_clients != 1:
        logger.warning(
            "local_only_hospital_com_min_clients_maior_que_1 valor=%s min_clients=%d — "
            "provável erro de configuração: treino local-only deveria usar min-clients=1",
            local_only_hospital, min_clients,
        )

    # Reflete FED_CFG.aggregation_strategy no registro — antes sempre "FedProx",
    # mesmo nas rodadas em que a agregação era, de fato, FedAvg puro (o próprio
    # FedProx do Flower não normaliza por τ) ou, agora, FedNova (achado
    # 2026-08-11, Seção~sec:fedprox-fednova-gap do rascunho do TCC). O termo
    # proximal do FedProx continua ativo no cliente nos dois casos — só a
    # agregação no servidor muda.
    _algorithm_label = "FedNova" if FED_CFG.aggregation_strategy == "fednova" else "FedProx"
    training_id = checkpoint_store.register_training(
        algorithm=_algorithm_label,
        log_file="",
        n_rounds_max=num_rounds,
        checkpoint_criterion=FED_CFG.checkpoint_criterion,
        partition_mode="natural",
        run_classification=run_classification,
        local_only_hospital=local_only_hospital,
        early_stop_enabled=FED_CFG.early_stop,
    )
    logger.info(
        "training_registered",
        extra={
            "training_id": training_id,
            "algorithm": "FedProx",
            "n_rounds_max": num_rounds,
            "run_classification": run_classification,
            "local_only_hospital": local_only_hospital,
            "early_stop_enabled": FED_CFG.early_stop,
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
        # vocab_json NÃO é montado aqui — _FitConfigMixin._inject_current_vocab() sobrescreve
        # com self.vocab (mutável, atualizado só uma vez em initialize_parameters(), antes da
        # Rodada 1 — ver strategy/core.py::_discover_and_curate_vocab). Essa closure não
        # conseguiria referenciar self.vocab de qualquer forma: a estratégia ainda não existe
        # no momento em que este lambda é definido. `recovered_vocab` aqui só alimenta o valor
        # INICIAL de self.vocab (constructor abaixo), não o vocab_json de cada rodada.
        on_fit_config_fn=lambda rnd: {
            "proximal_mu": proximal_mu,
            "local_epochs": local_epochs,
            "round": rnd,
        },
        # extract_rag_patterns: pede ao cliente pra extrair perfis prototípicos (RAG) só
        # na última rodada de verdade — caro pra repetir a cada round (forward com atenção
        # sobre o val_loader inteiro, uma vez por classe).
        # calibrate/calibration_method: pede ao cliente pra ajustar um calibrador local
        # (temperature ou isotonic, conforme FED_CFG.calibration_method) só na última
        # rodada de verdade — mesmo timing de extract_rag_patterns acima. Substitui a
        # calibração server-side (calibration_mixin.py), que exigiria um test_loader
        # centralizado nunca disponível no Caminho B por design de privacidade (ver
        # docs/Linha_do_Tempo_MOSAIC-FL.md, seção sobre F1 federado, 2026-07-07).
        #
        # "última rodada de verdade" == strategy.is_final_round(rnd) — com
        # FED_CFG.early_stop=False (default), é só `rnd >= num_rounds`, igual sempre foi.
        # Com FL_EARLY_STOP=true, pode disparar antes, no round agendado por
        # aggregate_evaluate() quando a convergência é detectada (achado 2026-08-01,
        # ver early_stop_server.py e strategy/core.py::is_final_round). A closure
        # referencia `strategy` mesmo definida antes da atribuição abaixo — só é
        # chamada depois, quando `strategy` já existe (late binding, mesmo padrão já
        # usado por vocab_json/_inject_current_vocab, comentário acima).
        on_evaluate_config_fn=lambda rnd: {
            "round": rnd,
            "extract_rag_patterns": strategy.is_final_round(rnd),
            "calibrate": strategy.is_final_round(rnd),
            "calibration_method": FED_CFG.calibration_method,
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
    # Server customizado só quando FL_EARLY_STOP=true (achado 2026-08-01, ver
    # early_stop_server.py) — None (default do ServerAppComponents) preserva
    # exatamente o Server padrão do flwr, loop fixo até num_rounds, comportamento
    # histórico intacto quando a flag está desligada.
    server = None
    if FED_CFG.early_stop:
        server = EarlyStoppingServer(client_manager=SimpleClientManager(), strategy=strategy)
        logger.info("early_stop_enabled — servidor customizado ativo (FL_EARLY_STOP=true)")

    # flwr/server/server.py::init_defaults() ignora `strategy` (com WARN) quando
    # `server` já foi passado — o Server customizado já carrega essa mesma
    # `strategy` internamente (linha acima), então passar os dois é só ruído no
    # log, sem efeito real (confirmado lendo o fonte). Evita o WARN redundante.
    return ServerAppComponents(
        server=server,
        strategy=None if server is not None else strategy,
        config=ServerConfig(num_rounds=num_rounds),
    )


# Entry point para: flwr run . <federation>
app = ServerApp(server_fn=_make_server_components)
