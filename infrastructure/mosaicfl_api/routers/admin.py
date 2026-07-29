"""Endpoints administrativos: /api/fl/status e /api/fl/reload. Startup checks."""
import logging
import os

from fastapi import APIRouter, Depends, HTTPException
from datetime import datetime

from integration.column_resolver import looks_like_valid_analyte_name

from .. import audit
from .. import state
from ..inference_engine import InferenceEngine
from ..schemas import (
    ClassWeightOverridesUpdate,
    FLStatus,
    OrchestrationConfigResponse,
    RagEvaluationEntry,
    RagEvaluationListResponse,
    RagEvaluationReportResponse,
    RagEvaluationScoreRequest,
    TrainingComparisonResponse,
    TrainingComparisonSide,
    TrainingResultSummary,
    TrainingResultsListResponse,
    TrainingRoundDetail,
    TrainingRoundsResponse,
    VocabAnomalyActionRequest,
    VocabAnomalyActionResponse,
    VocabAnomalyCorrectionRequest,
    VocabAnomalyCorrectionResponse,
    VocabAnomalyEntry,
    VocabAnomalyListResponse,
)
from ..security import _get_token_fingerprint

logger = logging.getLogger(__name__)

router = APIRouter()


async def startup_checks() -> None:
    """Valida configuração crítica — chamado no lifespan antes de aceitar tráfego."""
    from sqlalchemy import text as _text

    _env            = os.getenv("FL_ENV", "development").lower()
    _cors_origins   = os.getenv("FL_CORS_ORIGINS", "*").split(",")
    _pid_secret     = os.getenv("FL_PATIENT_ID_SECRET", "")
    _jwt_secret     = os.getenv("FL_JWT_SECRET", "")
    _jwt_pub_key    = ""
    _jwt_pub_file   = os.getenv("FL_JWT_PUBLIC_KEY_FILE", "")
    from pathlib import Path
    if _jwt_pub_file and Path(_jwt_pub_file).exists():
        _jwt_pub_key = Path(_jwt_pub_file).read_text(encoding="utf-8")

    errors: list[str] = []

    if not os.getenv("FL_DB_URL"):
        errors.append("FL_DB_URL não configurado")
    else:
        try:
            with state._db._engine.connect() as conn:
                conn.execute(_text("SELECT 1"))
        except Exception as exc:
            errors.append(f"Banco inacessível: {exc}")

    if _env == "production":
        if not _pid_secret:
            errors.append(
                "FL_PATIENT_ID_SECRET não configurado — patient_id seria armazenado em texto "
                "claro (viola LGPD Art. 13 §4º)"
            )
        if "*" in _cors_origins:
            errors.append(
                "FL_CORS_ORIGINS='*' não é permitido em FL_ENV=production — "
                "configure domínios explícitos"
            )
        if not (_jwt_secret or _jwt_pub_key):
            logger.warning(
                "startup_warning: FL_JWT_SECRET / FL_JWT_PUBLIC_KEY_FILE não configurados — "
                "autenticação valida apenas presença do token"
            )

    if errors:
        for msg in errors:
            logger.critical("startup_check_failed: %s", msg)
        raise RuntimeError(f"Configuração inválida para inicialização: {errors}")

    if not _pid_secret and _env != "production":
        logger.warning(
            "startup_warning: FL_PATIENT_ID_SECRET não configurado — "
            "patient_id será armazenado sem pseudonimização (apenas dev/testes)"
        )
    if "*" in _cors_origins and _env != "production":
        logger.warning(
            "startup_warning: FL_CORS_ORIGINS='*' — configure domínios explícitos antes de produção"
        )

    logger.warning(
        "startup_warning: rate_limiter=in_process — em produção com múltiplos workers "
        "(Gunicorn/Uvicorn), o limite é por processo e não global. "
        "Configure FL_REDIS_URL e substitua _SlidingWindowLimiter por fastapi-limiter + Redis."
    )
    logger.info(
        "startup_ok env=%s auth=%s jwt=%s pid_hash=%s cors_origins=%s",
        _env,
        os.getenv("FL_AUTH_REQUIRED", "true").lower() not in ("false", "0", "no"),
        bool(_jwt_secret or _jwt_pub_key),
        bool(_pid_secret),
        _cors_origins,
    )


@router.get("/api/fl/status", response_model=FLStatus)
async def fl_status():
    """Reflete o que a engine EM MEMÓRIA está de fato servindo — não um re-scan
    do diretório de checkpoints em arquivo. Achado 2026-07-28: um round_1.pt
    obsoleto (de 2026-07-06) sobrevivia em FL_CHECKPOINT_DIR e o status antigo
    sempre priorizava esse arquivo, mesmo quando a engine já tinha carregado um
    checkpoint muito mais recente via CheckpointStore (banco) — "1 round" ao lado
    de "Modelo pronto" mentia sobre o que estava realmente em produção.

    state._engine._checkpoint_round só é int em uma engine real (carregada via
    _load()/load_from_store()); em mocks de teste é um MagicMock, isinstance()
    falha e cai no fallback de arquivo (mantém os testes antigos válidos)."""
    engine = state._engine
    if engine is not None and isinstance(getattr(engine, "_checkpoint_round", None), int):
        return FLStatus(
            model_ready=True,
            checkpoint_path=str(engine._checkpoint_path) if engine._checkpoint_path else None,
            rounds_completed=engine._checkpoint_round,
            last_updated=engine._checkpoint_at,
        )

    ckpt = state._latest_checkpoint()
    rounds, last_updated = 0, None
    if ckpt:
        try:
            rounds = int(ckpt.stem.replace("round_", ""))
        except ValueError:
            pass
        last_updated = datetime.fromtimestamp(ckpt.stat().st_mtime).isoformat()
    return FLStatus(
        model_ready=ckpt is not None,
        checkpoint_path=str(ckpt) if ckpt else None,
        rounds_completed=rounds,
        last_updated=last_updated,
    )


@router.post("/api/fl/reload")
async def reload_model(fingerprint: str = Depends(_get_token_fingerprint)):
    """Força recarga do checkpoint mais recente com verificação de integridade."""
    ckpt = state._latest_checkpoint()
    if not ckpt:
        raise HTTPException(status_code=404, detail="nenhum checkpoint disponível")

    if not state._verify_checkpoint_integrity(ckpt):
        logger.error("checkpoint_integrity_failed path=%s", ckpt)
        raise HTTPException(
            status_code=500,
            detail=f"Checkpoint corrompido ou adulterado: {ckpt.name}. Retrainamento necessário.",
        )

    state._engine = InferenceEngine(
        checkpoint_path=ckpt,
        db_url=os.getenv("FL_DB_URL"),
    )
    logger.info("model_reloaded", extra={"checkpoint": str(ckpt)})
    audit.log_access("model_reload", token_fp=fingerprint, checkpoint=str(ckpt))
    return {"reloaded": True, "checkpoint": str(ckpt)}


@router.get("/api/admin/vocab-anomalies", response_model=VocabAnomalyListResponse)
async def list_vocab_anomalies(fingerprint: str = Depends(_get_token_fingerprint)):
    """Analitos do catálogo (knowledge.term_dictionary) cujo nome foge do padrão
    esperado — hoje só canonical puramente numérico (achado real: "183", aprovado
    automaticamente pela descoberta bidirecional de vocabulário em 2026-07-26,
    ver docs/Linha_do_Tempo_MOSAIC-FL.md). Heurística best-effort, não prova de
    erro — quem decide se corrige é quem revisa aqui.

    local_record_count só reflete o banco que ESTA instância da API enxerga (um
    hospital, por design de privacidade federada) — pode ser 0 mesmo quando o
    analito é legítimo em outro hospital.
    """
    from sqlalchemy import text as _text

    try:
        with state._db._engine.connect() as conn:
            rows = conn.execute(_text("""
                SELECT td.canonical, td.source, td.active, td.created_at,
                       ar.ref_low, ar.ref_high, ar.n_hospitals
                FROM knowledge.term_dictionary td
                LEFT JOIN knowledge.analyte_references ar
                       ON ar.canonical = td.canonical AND ar.sex IS NULL
                WHERE td.term_type = 'analyte'
                ORDER BY td.created_at DESC
            """)).fetchall()
    except Exception as exc:
        logger.warning("vocab_anomalies_query_error error=%s", exc)
        raise HTTPException(status_code=503, detail=f"Não foi possível consultar o catálogo: {exc}")

    anomalies: list[VocabAnomalyEntry] = []
    for r in rows:
        if looks_like_valid_analyte_name(r.canonical):
            continue
        try:
            with state._db._engine.connect() as conn:
                n = conn.execute(_text(
                    "SELECT count(*) FROM metrics.exam_records WHERE analyte = :c"
                ), {"c": r.canonical}).scalar()
        except Exception:
            n = 0
        anomalies.append(VocabAnomalyEntry(
            canonical=r.canonical,
            source=r.source,
            active=r.active,
            created_at=r.created_at.isoformat() if r.created_at else None,
            reason="canonical puramente numérico — não parece nome de exame",
            ref_low=r.ref_low,
            ref_high=r.ref_high,
            n_hospitals=r.n_hospitals,
            local_record_count=int(n or 0),
        ))
    return VocabAnomalyListResponse(anomalies=anomalies)


@router.post("/api/admin/vocab-anomalies/{canonical}", response_model=VocabAnomalyActionResponse)
async def set_vocab_anomaly_active(
    canonical: str,
    body: VocabAnomalyActionRequest,
    fingerprint: str = Depends(_get_token_fingerprint),
):
    """Ativa/desativa um canonical em knowledge.term_dictionary — correção manual
    depois de revisar em /vocab-anomalies. Desativar (active=FALSE) remove o
    analito de build_standard_vocab() sem apagar o histórico (exam_records e
    analyte_references ficam intocados, reversível a qualquer momento reativando)."""
    from sqlalchemy import text as _text

    try:
        with state._db._engine.connect() as conn:
            result = conn.execute(_text("""
                UPDATE knowledge.term_dictionary SET active = :active
                WHERE term_type = 'analyte' AND canonical = :canonical
            """), {"active": body.active, "canonical": canonical})
            conn.commit()
    except Exception as exc:
        logger.error("vocab_anomaly_update_error canonical=%s error=%s", canonical, exc)
        raise HTTPException(status_code=503, detail=f"Não foi possível atualizar o catálogo: {exc}")

    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail=f"canonical={canonical!r} não encontrado")

    logger.info("vocab_anomaly_corrected canonical=%s active=%s", canonical, body.active)
    audit.log_access(
        "vocab_anomaly_correction", token_fp=fingerprint,
        canonical=canonical, active=body.active,
    )
    return VocabAnomalyActionResponse(canonical=canonical, active=body.active)


@router.post("/api/admin/vocab-anomalies/{canonical}/correct", response_model=VocabAnomalyCorrectionResponse)
async def correct_vocab_anomaly(
    canonical: str,
    body: VocabAnomalyCorrectionRequest,
    fingerprint: str = Depends(_get_token_fingerprint),
):
    """Renomeia um canonical errado pro nome certo — equivalente web de
    correct_term() (integration/term_manager/pending_workflow.py), estendido
    pra também corrigir metrics.exam_records (a CLI só corrige o catálogo,
    nunca tocou nos dados já carregados).

    Dois casos:
      - new_canonical NÃO existe ainda: rename simples (UPDATE canonical em
        term_dictionary + analyte_references).
      - new_canonical JÁ existe (caso real: "183" -> "AMILASE", que já tinha
        13.189 registros legítimos do BPSP): funde — "183" vira alias de
        AMILASE, a linha de analyte_references de "183" é removida (redundante,
        AMILASE já tem a própria), exam_records migra pro nome certo. Mesmo
        padrão da migration 026, disponível aqui pra qualquer caso futuro sem
        precisar escrever uma migration nova toda vez."""
    from sqlalchemy import text as _text

    new_canonical = body.correct_canonical.strip().upper()
    if not new_canonical:
        raise HTTPException(status_code=422, detail="correct_canonical não pode ser vazio")
    if new_canonical == canonical:
        raise HTTPException(status_code=422, detail="correct_canonical é igual ao canonical atual")

    try:
        with state._db._engine.begin() as conn:
            exists = conn.execute(_text("""
                SELECT 1 FROM knowledge.term_dictionary
                WHERE term_type = 'analyte' AND canonical = :new LIMIT 1
            """), {"new": new_canonical}).fetchone()
            merged = exists is not None

            exam_result = conn.execute(_text("""
                UPDATE metrics.exam_records SET analyte = :new WHERE analyte = :old
            """), {"new": new_canonical, "old": canonical})

            if merged:
                conn.execute(_text("""
                    DELETE FROM knowledge.term_dictionary
                    WHERE term_type = 'analyte' AND canonical = :old
                """), {"old": canonical})
                conn.execute(_text("""
                    INSERT INTO knowledge.term_dictionary (term_type, canonical, alias, source, active)
                    VALUES ('analyte', :new, :old, 'MANUAL_CORRECTION', TRUE)
                    ON CONFLICT (term_type, canonical, alias) DO NOTHING
                """), {"new": new_canonical, "old": canonical})
                conn.execute(_text("""
                    DELETE FROM knowledge.analyte_references WHERE canonical = :old
                """), {"old": canonical})
            else:
                conn.execute(_text("""
                    UPDATE knowledge.term_dictionary SET canonical = :new
                    WHERE term_type = 'analyte' AND canonical = :old
                """), {"new": new_canonical, "old": canonical})
                conn.execute(_text("""
                    UPDATE knowledge.analyte_references SET canonical = :new
                    WHERE canonical = :old
                """), {"new": new_canonical, "old": canonical})
    except Exception as exc:
        logger.error(
            "vocab_anomaly_correction_error old=%s new=%s error=%s", canonical, new_canonical, exc,
        )
        raise HTTPException(status_code=503, detail=f"Não foi possível corrigir o catálogo: {exc}")

    logger.info(
        "vocab_anomaly_renamed old=%s new=%s merged=%s exam_records_updated=%d",
        canonical, new_canonical, merged, exam_result.rowcount,
    )
    audit.log_access(
        "vocab_anomaly_correction_rename", token_fp=fingerprint,
        old_canonical=canonical, new_canonical=new_canonical, merged=merged,
    )
    return VocabAnomalyCorrectionResponse(
        old_canonical=canonical, new_canonical=new_canonical,
        merged=merged, exam_records_updated=exam_result.rowcount,
    )


@router.get("/api/admin/orchestration-config", response_model=OrchestrationConfigResponse)
async def get_orchestration_config(fingerprint: str = Depends(_get_token_fingerprint)):
    """Peso de classe explícito (cost-sensitive learning, Strategy pattern em
    mosaicfl.core.class_weighting) — lido de clinical.fl_orchestration_config
    (migration 028), o mesmo canal que o servidor de orquestração FL já usa pra
    empurrar proximal_mu/stop pro cliente a cada rodada, idêntico nos dois
    hospitais. Ver docs/pesquisa_baseline_implementacao_fontes_bibliograficas.md,
    seção 14."""
    from sqlalchemy import text as _text

    from mosaicfl.core.config import FED_CFG, MODEL_CFG

    try:
        with state._db._engine.connect() as conn:
            row = conn.execute(_text(
                "SELECT class_weight_overrides_json "
                "FROM clinical.fl_orchestration_config WHERE id = 'current'"
            )).mappings().first()
    except Exception as exc:
        logger.warning("orchestration_config_query_error error=%s", exc)
        raise HTTPException(status_code=503, detail=f"Não foi possível consultar a configuração: {exc}")

    overrides = (row["class_weight_overrides_json"] if row else None) or {}
    return OrchestrationConfigResponse(
        class_weight_overrides=overrides,
        class_labels=list(MODEL_CFG.class_labels),
        class_weight_clamp=FED_CFG.class_weight_clamp,
    )


@router.post("/api/admin/orchestration-config/class-weights", response_model=OrchestrationConfigResponse)
async def set_class_weight_overrides(
    body: ClassWeightOverridesUpdate,
    fingerprint: str = Depends(_get_token_fingerprint),
):
    """Substitui o conjunto inteiro de overrides de peso de classe (não é merge
    parcial — reflete exatamente o que a tela mandou). Classe fora daqui cai em
    class_balanced (frequência local). Valor só tem efeito real na 1ª rodada de
    um treino novo (FedProxClient._ensure_data carrega o criterion uma vez só) —
    não afeta um treino já em andamento retroativamente."""
    from sqlalchemy import text as _text

    from mosaicfl.core.class_weighting import validate_overrides
    from mosaicfl.core.config import MODEL_CFG

    try:
        validate_overrides(body.overrides, MODEL_CFG.class_labels)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    import json as _json
    overrides_json = _json.dumps(body.overrides) if body.overrides else None

    try:
        with state._db._engine.begin() as conn:
            conn.execute(_text("""
                UPDATE clinical.fl_orchestration_config
                SET class_weight_overrides_json = cast(:overrides AS jsonb), updated_at = now()
                WHERE id = 'current'
            """), {"overrides": overrides_json})
    except Exception as exc:
        logger.error("orchestration_config_update_error error=%s", exc)
        raise HTTPException(status_code=503, detail=f"Não foi possível gravar a configuração: {exc}")

    logger.info("class_weight_overrides_updated overrides=%s", body.overrides)
    audit.log_access(
        "class_weight_overrides_update", token_fp=fingerprint, overrides=body.overrides,
    )
    from mosaicfl.core.config import FED_CFG
    return OrchestrationConfigResponse(
        class_weight_overrides=body.overrides,
        class_labels=list(MODEL_CFG.class_labels),
        class_weight_clamp=FED_CFG.class_weight_clamp,
    )


# LEFT JOIN com fl_checkpoints pra expor checkpoint_round — sanity-check direto
# na tela do achado #checkpoint (2026-07-28): fl_checkpoints.round DEVE bater com
# fl_trainings.best_round; ver project_bug_checkpoint_nao_era_melhor_rodada.
_TRAINING_SUMMARY_COLUMNS = (
    "t.id, t.algorithm, t.run_classification, t.partition_mode, t.status, t.started_at, t.completed_at, "
    "t.n_rounds_done, t.best_round, t.best_accuracy, t.converged, t.macro_f1, t.macro_auc, t.ece, t.ece_pre, "
    "t.total_duration_s, t.dp_noise_multiplier, t.dp_noise_strategy, t.dp_epsilon_simple, t.dp_epsilon_rdp, "
    "t.is_active_model, c.round AS checkpoint_round"
)
_TRAINING_SUMMARY_FROM = (
    "metrics.fl_trainings t LEFT JOIN metrics.fl_checkpoints c ON c.training_id = t.id"
)


def _row_to_training_summary(r) -> TrainingResultSummary:
    mismatch = (
        r.best_round is not None and r.checkpoint_round is not None
        and r.best_round != r.checkpoint_round
    )
    return TrainingResultSummary(
        id=r.id, algorithm=r.algorithm, run_classification=r.run_classification,
        partition_mode=r.partition_mode, status=r.status,
        started_at=r.started_at.isoformat() if r.started_at else None,
        completed_at=r.completed_at.isoformat() if r.completed_at else None,
        n_rounds_done=r.n_rounds_done, best_round=r.best_round, best_accuracy=r.best_accuracy,
        converged=r.converged, macro_f1=r.macro_f1, macro_auc=r.macro_auc, ece=r.ece,
        ece_pre=r.ece_pre, total_duration_s=r.total_duration_s,
        dp_noise_multiplier=r.dp_noise_multiplier, dp_noise_strategy=r.dp_noise_strategy,
        dp_epsilon_simple=r.dp_epsilon_simple, dp_epsilon_rdp=r.dp_epsilon_rdp,
        is_active_model=r.is_active_model,
        checkpoint_round=r.checkpoint_round, checkpoint_mismatch=mismatch,
    )


@router.get("/api/admin/fl-training-results", response_model=TrainingResultsListResponse)
async def list_training_results(fingerprint: str = Depends(_get_token_fingerprint)):
    """Lista todos os treinamentos federados (metrics.fl_trainings), mais recente
    primeiro — visão resumida pra tela /fl-training-results. Ver
    docs/pesquisa_baseline_implementacao_fontes_bibliograficas.md e a linha do
    tempo pra interpretação de cada achado por treino."""
    from sqlalchemy import text as _text

    try:
        with state._db._engine.connect() as conn:
            rows = conn.execute(_text(
                f"SELECT {_TRAINING_SUMMARY_COLUMNS} FROM {_TRAINING_SUMMARY_FROM} ORDER BY t.id DESC"
            )).fetchall()
    except Exception as exc:
        logger.warning("training_results_query_error error=%s", exc)
        raise HTTPException(status_code=503, detail=f"Não foi possível consultar os treinamentos: {exc}")

    return TrainingResultsListResponse(trainings=[_row_to_training_summary(r) for r in rows])


@router.get("/api/admin/fl-training-results/{training_id}/rounds", response_model=TrainingRoundsResponse)
async def get_training_rounds(training_id: int, fingerprint: str = Depends(_get_token_fingerprint)):
    """Histórico rodada-a-rodada de um treinamento — per_class_f1 (agregado) e
    per_client_f1 (por hospital, antes da agregação, migration 027) pra avaliar
    estabilidade/diluição, mesma análise feita manualmente pra decidir se um
    ajuste de peso de classe funcionou de verdade (ver seção 13/14 do doc de
    pesquisa)."""
    from sqlalchemy import text as _text

    from mosaicfl.core.config import MODEL_CFG

    try:
        with state._db._engine.connect() as conn:
            best_round_row = conn.execute(_text(
                "SELECT best_round FROM metrics.fl_trainings WHERE id = :id"
            ), {"id": training_id}).mappings().first()
            checkpoint_row = conn.execute(_text(
                "SELECT round, evaluation_json FROM metrics.fl_checkpoints WHERE training_id = :id"
            ), {"id": training_id}).mappings().first()
            rows = conn.execute(_text("""
                SELECT round, accuracy, f1_macro, per_class_f1, per_client_f1_json
                FROM metrics.fl_round_history WHERE training_id = :id ORDER BY round
            """), {"id": training_id}).fetchall()
    except Exception as exc:
        logger.warning("training_rounds_query_error training_id=%s error=%s", training_id, exc)
        raise HTTPException(status_code=503, detail=f"Não foi possível consultar as rodadas: {exc}")

    if not rows:
        raise HTTPException(status_code=404, detail=f"training_id={training_id!r} sem histórico de rodadas")

    rounds = [
        TrainingRoundDetail(
            round=r.round, accuracy=r.accuracy, f1_macro=r.f1_macro,
            per_class_f1=r.per_class_f1, per_client_f1=r.per_client_f1_json,
        )
        for r in rows
    ]
    return TrainingRoundsResponse(
        training_id=training_id,
        class_labels=list(MODEL_CFG.class_labels),
        best_round=best_round_row["best_round"] if best_round_row else None,
        checkpoint_round=checkpoint_row["round"] if checkpoint_row else None,
        evaluation_json=checkpoint_row["evaluation_json"] if checkpoint_row else None,
        rounds=rounds,
    )


@router.get("/api/admin/fl-training-results/compare", response_model=TrainingComparisonResponse)
async def compare_training_results(fingerprint: str = Depends(_get_token_fingerprint)):
    """Compara o treinamento MAIS RECENTE contra o de MELHOR macro_f1 até hoje —
    mesma comparação feita manualmente pra avaliar se um ajuste (peso de classe,
    ruído seletivo etc.) melhorou ou piorou em relação ao que já se tinha.

    Restrito a partition_mode='natural' (achado 2026-07-29): sem esse filtro, um
    treino do Caminho A com FL_PARTITION_MODE=iid_simulado (dado artificialmente
    embaralhado/balanceado, muito mais fácil de acertar macro_f1 do que o non-IID
    real de BPSP/HSL) podia vencer a comparação e parecer "o melhor modelo" —
    maçã com laranja, não avalia a mesma coisa. Ver seção "Paridade Caminho A x B"
    em docs/Metodologia_Executada_MOSAIC-FL.tex."""
    from sqlalchemy import text as _text

    from mosaicfl.core.config import MODEL_CFG

    try:
        with state._db._engine.connect() as conn:
            latest_row = conn.execute(_text(
                f"SELECT {_TRAINING_SUMMARY_COLUMNS} FROM {_TRAINING_SUMMARY_FROM} "
                "WHERE t.status = 'completed' AND t.partition_mode = 'natural' "
                "ORDER BY t.id DESC LIMIT 1"
            )).fetchone()
            best_row = conn.execute(_text(
                f"SELECT {_TRAINING_SUMMARY_COLUMNS} FROM {_TRAINING_SUMMARY_FROM} "
                "WHERE t.status = 'completed' AND t.macro_f1 IS NOT NULL "
                "AND t.partition_mode = 'natural' "
                "ORDER BY t.macro_f1 DESC LIMIT 1"
            )).fetchone()

            def _per_class_f1_at_best_round(training_id: int, best_round: int):
                if training_id is None or best_round is None:
                    return None
                row = conn.execute(_text(
                    "SELECT per_class_f1 FROM metrics.fl_round_history "
                    "WHERE training_id = :tid AND round = :rnd"
                ), {"tid": training_id, "rnd": best_round}).mappings().first()
                return row["per_class_f1"] if row else None

            latest_pcf1 = _per_class_f1_at_best_round(latest_row.id, latest_row.best_round) if latest_row else None
            best_pcf1 = _per_class_f1_at_best_round(best_row.id, best_row.best_round) if best_row else None
    except Exception as exc:
        logger.warning("training_comparison_query_error error=%s", exc)
        raise HTTPException(status_code=503, detail=f"Não foi possível montar a comparação: {exc}")

    return TrainingComparisonResponse(
        class_labels=list(MODEL_CFG.class_labels),
        latest=TrainingComparisonSide(
            training=_row_to_training_summary(latest_row), per_class_f1=latest_pcf1,
        ) if latest_row else None,
        best=TrainingComparisonSide(
            training=_row_to_training_summary(best_row), per_class_f1=best_pcf1,
        ) if best_row else None,
    )


_RAG_EVAL_COLUMNS = (
    "id, patient_id_hash, predicted_label, risk_score, justificativa, fontes_json, "
    "llm_backend, llm_model_used, llm_was_fallback, alucinacao_detectada, confiavel, "
    "likert_score, llm_judge_score, llm_judge_rationale, evaluator, checkpoint_round, created_at"
)


def _row_to_rag_evaluation(r) -> RagEvaluationEntry:
    return RagEvaluationEntry(
        id=r.id, patient_id_hash=r.patient_id_hash, predicted_label=r.predicted_label,
        risk_score=r.risk_score, justificativa=r.justificativa, fontes_json=r.fontes_json,
        llm_backend=r.llm_backend, llm_model_used=r.llm_model_used,
        llm_was_fallback=r.llm_was_fallback, alucinacao_detectada=r.alucinacao_detectada,
        confiavel=r.confiavel, likert_score=r.likert_score, llm_judge_score=r.llm_judge_score,
        llm_judge_rationale=r.llm_judge_rationale, evaluator=r.evaluator,
        checkpoint_round=r.checkpoint_round,
        created_at=r.created_at.isoformat() if r.created_at else None,
    )


@router.get("/api/admin/rag-evaluations", response_model=RagEvaluationListResponse)
async def list_rag_evaluations(
    pending_only: bool = True,
    fingerprint: str = Depends(_get_token_fingerprint),
):
    """Amostras geradas por scripts/rag_likert_evaluation.py --generate.
    pending_only=True (default): só as que ainda não têm likert_score (nota
    HUMANA) — é a fila de trabalho de quem vai avaliar. pending_only=False:
    todas, incluindo já avaliadas (revisão/auditoria)."""
    from sqlalchemy import text as _text

    where = "likert_score IS NULL" if pending_only else "1=1"
    try:
        with state._db._engine.connect() as conn:
            rows = conn.execute(_text(
                f"SELECT {_RAG_EVAL_COLUMNS} FROM metrics.rag_likert_evaluations "
                f"WHERE {where} ORDER BY id"
            )).fetchall()
    except Exception as exc:
        logger.warning("rag_evaluations_query_error error=%s", exc)
        raise HTTPException(status_code=503, detail=f"Não foi possível consultar as avaliações: {exc}")

    return RagEvaluationListResponse(evaluations=[_row_to_rag_evaluation(r) for r in rows])


@router.post("/api/admin/rag-evaluations/{evaluation_id}/score")
async def score_rag_evaluation(
    evaluation_id: int,
    body: RagEvaluationScoreRequest,
    fingerprint: str = Depends(_get_token_fingerprint),
):
    """Registra a nota HUMANA (likert_score) de uma amostra já gerada — nunca
    inferida automaticamente (ver llm_judge_score, métrica complementar
    separada, achado 2026-07-29)."""
    from sqlalchemy import text as _text

    try:
        with state._db._engine.connect() as conn:
            result = conn.execute(_text(
                "UPDATE metrics.rag_likert_evaluations "
                "SET likert_score = :score, evaluator = :evaluator "
                "WHERE id = :id"
            ), {"score": body.likert_score, "evaluator": body.evaluator, "id": evaluation_id})
            conn.commit()
    except Exception as exc:
        logger.error("rag_evaluation_score_error id=%s error=%s", evaluation_id, exc)
        raise HTTPException(status_code=503, detail=f"Não foi possível salvar a nota: {exc}")

    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail=f"Avaliação id={evaluation_id} não encontrada")

    audit.log_access("rag_evaluation_scored", token_fp=fingerprint, evaluation_id=evaluation_id)
    return {"id": evaluation_id, "likert_score": body.likert_score}


@router.get("/api/admin/rag-evaluations/report", response_model=RagEvaluationReportResponse)
async def rag_evaluations_report(fingerprint: str = Depends(_get_token_fingerprint)):
    """Mesmo resumo de `rag_likert_evaluation.py --report`, via API — as duas
    métricas (humana e automática) sempre lado a lado, nunca uma no lugar da
    outra."""
    from sqlalchemy import text as _text

    try:
        with state._db._engine.connect() as conn:
            rows = conn.execute(_text(
                "SELECT likert_score, llm_judge_score, alucinacao_detectada, confiavel "
                "FROM metrics.rag_likert_evaluations"
            )).fetchall()
    except Exception as exc:
        logger.warning("rag_evaluations_report_error error=%s", exc)
        raise HTTPException(status_code=503, detail=f"Não foi possível montar o resumo: {exc}")

    n_total = len(rows)
    if n_total == 0:
        return RagEvaluationReportResponse(n_total=0, n_hallucination=0, n_confiavel=0, n_human_scored=0, n_judge_scored=0, n_both_scored=0)

    n_hallucination = sum(1 for r in rows if r.alucinacao_detectada)
    n_confiavel = sum(1 for r in rows if r.confiavel)

    human = [r for r in rows if r.likert_score is not None]
    human_dist: dict[int, int] = {}
    for r in human:
        human_dist[r.likert_score] = human_dist.get(r.likert_score, 0) + 1
    human_pct_ge4 = 100 * sum(1 for r in human if r.likert_score >= 4) / len(human) if human else None

    judged = [r for r in rows if r.llm_judge_score is not None]
    judge_dist: dict[int, int] = {}
    for r in judged:
        judge_dist[r.llm_judge_score] = judge_dist.get(r.llm_judge_score, 0) + 1
    judge_pct_ge4 = 100 * sum(1 for r in judged if r.llm_judge_score >= 4) / len(judged) if judged else None

    both = [r for r in rows if r.likert_score is not None and r.llm_judge_score is not None]
    agreement_exact = 100 * sum(1 for r in both if r.likert_score == r.llm_judge_score) / len(both) if both else None
    agreement_ge4 = (
        100 * sum(1 for r in both if (r.likert_score >= 4) == (r.llm_judge_score >= 4)) / len(both)
        if both else None
    )

    return RagEvaluationReportResponse(
        n_total=n_total, n_hallucination=n_hallucination, n_confiavel=n_confiavel,
        n_human_scored=len(human), human_pct_ge4=human_pct_ge4, human_distribution=human_dist,
        n_judge_scored=len(judged), judge_pct_ge4=judge_pct_ge4, judge_distribution=judge_dist,
        n_both_scored=len(both), agreement_exact_pct=agreement_exact, agreement_ge4_pct=agreement_ge4,
    )
