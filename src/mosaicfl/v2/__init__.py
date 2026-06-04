"""
mosaicfl.v2 — Integração com dataset real e clientes federados externos.

Melhorias em relação à v1:
  - Masked Mean Pooling no modelo BEHRT
  - get_parameters com parâmetros treináveis apenas
  - ConvergenceTracker integrado ao servidor
  - RAG type-safe (generate_justification retorna 3 valores)
  - data_loader flexível (CSV, PostgreSQL, MySQL, SQLite, etc.)

Ponto de entrada: python run_v2.py
"""
