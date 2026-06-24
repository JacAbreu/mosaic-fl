"""
service.py — shim de compatibilidade para imports legados.

A aplicação real está em app.py.
Os endpoints estão em routers/ (prediction, patients, admin).
O estado mutável está em state.py.

Manter este módulo garante que imports externos que fazem
    from infrastructure.mosaicfl_api.service import app
ou
    import infrastructure.mosaicfl_api.service as svc; svc.app
continuem funcionando sem alteração.
"""
from .app import app   # noqa: F401
from . import state    # noqa: F401  — para acesso via svc.state em scripts legados
