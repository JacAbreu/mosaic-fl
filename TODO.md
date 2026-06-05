# TODO — Roadmap de Produção MOSAIC-FL

Lista de tarefas para evolução do projeto além do escopo do TCC.

## Dados e Integração

- [ ] Integração HL7 FHIR com EPR dos hospitais
- [ ] Conector genérico para prontuários eletrônicos brasileiros (MV, Tasy, Soul MV)

## Segurança e Privacidade

- [ ] TLS mútuo (mTLS) entre servidor e clientes Flower
- [ ] Differential Privacy nos pesos (Gaussian mechanism, ε-δ DP)
- [ ] Auditoria de acesso e rastreabilidade conforme LGPD
- [ ] Consentimento informado e designação de DPO

## Modelo

- [ ] Fine-tuning em corpus clínico brasileiro (MIMIC-BR ou equivalente)
- [ ] Substituir DistilGPT-2 por LLM em português (Maritaca, Llama-PT)
- [ ] Avaliação com AUC-ROC, sensibilidade e especificidade em estudo retrospectivo

## Infraestrutura

- [ ] Chamadas gRPC diretas do scheduler para o servidor Flower (hoje o scheduler é apenas supervisor)
- [ ] Message broker (RabbitMQ ou Redis) para orquestração de rounds
- [ ] Integração com `fl.server.Driver` do Flower SDK para controle programático
- [ ] Monitoramento com Prometheus + Grafana para métricas de treino federado

## Regulatório

- [ ] Submissão ANVISA como Software como Dispositivo Médico (SaMD) Classe II/III
- [ ] Validação clínica prospectiva com parecer de comitê de ética (CEP/CONEP)
