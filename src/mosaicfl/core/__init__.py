"""
mosaicfl.core — Domínio puro do MosaicFL.

Contém toda a lógica de negócio independente de framework de deployment:
modelo BEHRT, cliente FedProx, convergência, pré-processamento, RAG,
interpretabilidade e utilitários de agregação federada.

Importado por todos os adapters (infrastructure/ para produção,
experiments/ para pesquisa local).
"""
