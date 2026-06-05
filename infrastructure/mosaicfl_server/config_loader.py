"""
config_loader.py
Carregadores de configuração de runtime para o orquestrador MOSAIC-FL.

Uso:
    loader = get_config_loader()
    config = loader.load(round_num=3)
    # config: {"proximal_mu": 0.01, "pause_seconds": 0.0, "stop": False}

Backends suportados (FL_CONFIG_BACKEND):
    chroma  — ChromaDB (padrão; já usado pelo RAG do projeto)
    file    — JSON local em logs/runtime_config.json (fallback dev)

Para escrever uma nova config via ChromaDB:
    loader = ChromaDBConfigLoader()
    loader.write({"proximal_mu": 0.005, "stop": False})

Schema da collection ChromaDB:
    collection: "fl_orchestration_config"
    id fixo:    "runtime_config"
    metadata:   chaves do config (valores primitivos: str, int, float, bool)
"""
import json
import logging
import os
from pathlib import Path
from typing import Dict, Protocol

logger = logging.getLogger(__name__)

_CHROMA_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
_COLLECTION_NAME = "fl_orchestration_config"
_DOC_ID = "runtime_config"

_DEFAULTS: Dict = {
    "proximal_mu": None,      # None = não sobrescreve o valor da strategy
    "pause_seconds": 0.0,
    "stop": False,
}


class ConfigLoader(Protocol):
    def load(self, round_num: int) -> Dict:
        """Retorna config para o round. Retorna {} em caso de erro."""
        ...


class ChromaDBConfigLoader:
    """
    Lê e escreve config de runtime na collection ChromaDB existente do projeto.

    Não exige infraestrutura adicional: usa o mesmo PersistentClient
    já configurado em rag_system_v2.py (CHROMA_DB_PATH).
    """

    def __init__(self, db_path: str = _CHROMA_PATH) -> None:
        import chromadb
        self._client = chromadb.PersistentClient(path=db_path)
        self._collection = self._client.get_or_create_collection(_COLLECTION_NAME)

    def load(self, round_num: int) -> Dict:
        try:
            result = self._collection.get(ids=[_DOC_ID])
            if not result["metadatas"] or result["metadatas"][0] is None:
                return {}
            raw = result["metadatas"][0]
            return self._cast(raw)
        except Exception as e:
            logger.warning("config_load_error", extra={"backend": "chroma", "round": round_num, "error": str(e)})
            return {}

    def write(self, config: Dict) -> None:
        """
        Persiste config no ChromaDB.

        Exemplo:
            loader.write({"proximal_mu": 0.005, "stop": False})
        """
        # ChromaDB metadata só aceita str/int/float/bool
        metadata = {k: v for k, v in config.items() if isinstance(v, (str, int, float, bool))}
        try:
            self._collection.upsert(
                ids=[_DOC_ID],
                documents=["runtime_config"],
                metadatas=[metadata],
            )
            logger.info("config_written", extra={"backend": "chroma", "keys": list(metadata.keys())})
        except Exception as e:
            logger.error("config_write_error", extra={"backend": "chroma", "error": str(e)})

    def clear(self) -> None:
        """Remove o documento de config (volta para defaults da strategy)."""
        try:
            self._collection.delete(ids=[_DOC_ID])
            logger.info("config_cleared", extra={"backend": "chroma"})
        except Exception:
            pass

    @staticmethod
    def _cast(raw: Dict) -> Dict:
        """Converte strings do metadata ChromaDB para os tipos corretos."""
        result: Dict = {}
        for key, value in raw.items():
            if key == "stop":
                result[key] = str(value).lower() in ("true", "1", "yes")
            elif key in ("proximal_mu", "pause_seconds"):
                try:
                    result[key] = float(value)
                except (ValueError, TypeError):
                    pass
            else:
                result[key] = value
        return result


class FileConfigLoader:
    """
    Lê config de um arquivo JSON local.

    Útil para desenvolvimento local sem ChromaDB.
    Arquivo: logs/runtime_config.json
    """

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or (Path(os.getenv("FL_LOG_DIR", "logs")) / "runtime_config.json")

    def load(self, round_num: int) -> Dict:
        if not self._path.exists():
            return {}
        try:
            with open(self._path, encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning("config_load_error", extra={"backend": "file", "round": round_num, "error": str(e)})
            return {}

    def write(self, config: Dict) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        logger.info("config_written", extra={"backend": "file", "path": str(self._path)})

    def clear(self) -> None:
        self._path.unlink(missing_ok=True)


def get_config_loader() -> ChromaDBConfigLoader | FileConfigLoader:
    """
    Instancia o loader correto baseado em FL_CONFIG_BACKEND.

    FL_CONFIG_BACKEND=chroma  → ChromaDBConfigLoader (padrão)
    FL_CONFIG_BACKEND=file    → FileConfigLoader
    """
    backend = os.getenv("FL_CONFIG_BACKEND", "chroma").lower()
    if backend == "file":
        logger.info("config_loader_selected", extra={"backend": "file"})
        return FileConfigLoader()
    logger.info("config_loader_selected", extra={"backend": "chroma", "db_path": _CHROMA_PATH})
    return ChromaDBConfigLoader()
