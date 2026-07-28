"""
local_db_source.py — Lê class_weight_overrides_json do banco LOCAL desta máquina
(clinical.fl_orchestration_config), 2º nível de prioridade em FedProxClient — entre
o valor empurrado pelo servidor (compartilhado, idêntico nos dois hospitais) e o
fallback de env var (FED_CFG.class_weight_overrides).

Cada hospital tem seu PRÓPRIO banco (dado clínico nunca sai do hospital, por
desenho — ver docs/pesquisa_baseline_implementacao_fontes_bibliograficas.md, seção
14). Ler daqui é o que permite BPSP e HSL terem pesos DIFERENTES entre si, sem
precisar de nenhum mecanismo novo no protocolo Flower — o cliente já tem acesso ao
seu próprio banco via FL_DB_URL, usado pra tudo mais (SGBDDataSource etc.). Editável
via scripts/set_class_weight_overrides.py (make server-set-class-weights /
make client-set-class-weights) ou pela tela /class-weights, rodada localmente em
cada máquina.
"""
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


def load_local_overrides(db_url: str) -> Optional[Dict[str, float]]:
    """Overrides gravados no banco local desta máquina, ou None se não houver
    (linha ausente, coluna NULL, db_url vazio, ou qualquer erro de conexão/consulta
    — nunca propaga exceção; é um fallback opcional, não um requisito de boot)."""
    if not db_url:
        return None
    try:
        import sqlalchemy as sa
        engine = sa.create_engine(db_url)
        with engine.connect() as conn:
            row = conn.execute(sa.text(
                "SELECT class_weight_overrides_json FROM clinical.fl_orchestration_config WHERE id = 'current'"
            )).mappings().first()
        if not row or row["class_weight_overrides_json"] is None:
            return None
        return row["class_weight_overrides_json"]
    except Exception as e:
        logger.warning("local_class_weight_overrides_error error=%s", e)
        return None
