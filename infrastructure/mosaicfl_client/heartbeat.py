"""
heartbeat.py
Registra status do cliente para o scheduler monitorar.

Usa file locking (flock) para evitar race conditions quando múltiplos
clientes escrevem no mesmo registry simultaneamente.
"""
import fcntl
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

LOG_DIR = Path(os.getenv("FL_LOG_DIR", "logs"))
CLIENT_ID = os.getenv("FL_CLIENT_ID", "client_0")

# Timeout para adquirir lock (evita deadlock)
LOCK_TIMEOUT = 5.0


def write_heartbeat(status: str = "ready", registry_path: Optional[str] = None):
    """
    Escreve status no registry compartilhado de forma atômica.
    
    Usa file locking (flock) para garantir que apenas um cliente
    modifique o arquivo por vez, evitando corrupção de dados.
    """
    registry_file = Path(registry_path) if registry_path else LOG_DIR / "client_registry.json"
    registry_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Cria arquivo se não existir
    if not registry_file.exists():
        registry_file.write_text("{}", encoding="utf-8")
    
    try:
        # Abre em modo leitura/escrita
        with open(registry_file, "r+", encoding="utf-8") as f:
            # Adquire lock exclusivo (bloqueia até conseguir)
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            
            try:
                # Lê conteúdo atual
                content = f.read().strip()
                registry = json.loads(content) if content else {}
                
                # Atualiza dados deste cliente
                registry[CLIENT_ID] = {
                    "last_seen": datetime.now().timestamp(),
                    "status": status,
                    "client_id": CLIENT_ID,
                }
                
                # Volta ao início do arquivo e trunca
                f.seek(0)
                f.truncate()
                json.dump(registry, f, indent=2, ensure_ascii=False)
                
            finally:
                # Sempre libera o lock
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                
    except json.JSONDecodeError as e:
        logger.error(f"Erro ao decodificar JSON do registry: {e}")
        # Backup e recria com JSON válido contendo este cliente.
        backup = registry_file.with_suffix(".json.bak")
        registry_file.rename(backup)
        logger.info(f"Registry corrompido salvo em {backup}. Criando novo.")
        new_registry = {
            CLIENT_ID: {
                "last_seen": datetime.now().timestamp(),
                "status": status,
                "client_id": CLIENT_ID,
            }
        }
        registry_file.write_text(
            json.dumps(new_registry, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        
    except PermissionError as e:
        logger.error(f"Permissão negada ao acessar registry: {e}")
        raise
        
    except Exception as e:
        logger.error(f"Erro inesperado ao escrever heartbeat: {e}")
        raise


def read_heartbeat(client_id: Optional[str] = None, registry_path: Optional[str] = None) -> Optional[dict]:
    """
    Lê o heartbeat de um cliente específico ou de todos.
    
    Args:
        client_id: ID do cliente (se None, retorna todos)
        registry_path: Caminho alternativo para o registry
        
    Returns:
        dict com dados do heartbeat ou None se não encontrado
    """
    registry_file = Path(registry_path) if registry_path else LOG_DIR / "client_registry.json"
    
    if not registry_file.exists():
        return None
    
    try:
        with open(registry_file, "r", encoding="utf-8") as f:
            # Usa lock compartilhado para leitura (não bloqueia escritas)
            fcntl.flock(f.fileno(), fcntl.LOCK_SH)
            try:
                content = f.read().strip()
                if not content:
                    return None
                registry = json.loads(content)
                
                if client_id:
                    return registry.get(client_id)
                return registry
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Erro ao ler registry: {e}")
        return None


def is_client_alive(client_id: str, timeout_seconds: float = 600.0, registry_path: Optional[str] = None) -> bool:
    """
    Verifica se um cliente está "vivo" (último heartbeat dentro do timeout).
    
    Args:
        client_id: ID do cliente
        timeout_seconds: Timeout em segundos (padrão: 10 min)
        registry_path: Caminho alternativo para o registry
        
    Returns:
        True se o cliente está vivo, False caso contrário
    """
    heartbeat = read_heartbeat(client_id, registry_path)
    
    if not heartbeat:
        return False
    
    last_seen = heartbeat.get("last_seen")
    if not last_seen:
        return False
    
    from time import time
    elapsed = time() - last_seen
    return elapsed < timeout_seconds