# Motivações para o Enquadramento do MOSAIC-FL como AI-based CDSS

> Este documento registra a discussão que fundamentou a decisão de posicionar o MOSAIC-FL
> como um *Clinical Decision Support System* baseado em IA (AI-based CDSS), em vez de
> sistema autônomo de predição clínica.

---

## 1. O que é um CDSS e por que o MOSAIC-FL se encaixa nessa categoria

Existem dois tipos de sistemas de apoio à decisão clínica:

| Tipo | Base técnica | Exemplos |
|---|---|---|
| **CDSS tradicional** | Regras explícitas (if-then), árvores de decisão, protocolos codificados | Alerta de interação medicamentosa, calculadora de dose, checklist de sepse |
| **AI-based CDSS** | Machine learning, deep learning, NLP | MOSAIC-FL — FedProx + BEHRT + RAG |

O MOSAIC-FL é inequivocamente AI-based CDSS: usa rede neural transformer (SimplifiedBEHRT),
aprendizado federado (FedProx) e recuperação aumentada por embeddings (MiniLM + DistilGPT-2).

**Princípio central de design:** o MOSAIC-FL não substitui o julgamento clínico — apresenta
ao profissional de saúde uma distribuição de probabilidade sobre desfechos possíveis como
insumo para a tomada de decisão. A responsabilidade clínica permanece integralmente com o
profissional de saúde.

---

## 2. "Predição" não extrapola o limite do CDSS — depende do sujeito da frase

A palavra "predição" é comum na literatura de informática em saúde mesmo para sistemas
humano-no-loop. O problema não é a palavra — é o sujeito da frase:

| Formulação | Adequada para CDSS? |
|---|---|
| *"O sistema prediz que o paciente vai para UTI"* | Não — afirmação clínica de desfecho autônoma |
| *"O modelo estima probabilidade de desfecho"* | Sim — output matemático de um classificador |
| *"Estratificação de risco para suporte ao julgamento clínico"* | Sim — linguagem clínica padrão de CDSS |

O MOSAIC-FL entrega uma **distribuição de probabilidade sobre desfechos possíveis**, não uma
sentença clínica. O MС Dropout produz média e desvio padrão por classe — incerteza explícita
que o clínico interpreta no contexto do paciente real.

### Definição adotada no repositório

> **Módulo de Predição Federada para Possibilidades de Diagnóstico e Evoluções Clínicas**,
> *o modelo estima probabilidades de evoluções de quadros clínicos de acordo com as informações
> clínicas disponibilizadas, estratificando o risco*

---

## 3. Implicações regulatórias do enquadramento CDSS

| Instância | Implicação |
|---|---|
| **ANVISA (RDC 657/2022)** | AI-based CDSS com humano no loop tende à classificação SaMD Classe II (menor risco), não Classe III |
| **FDA (AI/ML-based SaMD)** | Distingue CDSS não-AI (geralmente isento de 510k) de AI/ML-based SaMD com humano no loop (menor supervisão que sistemas autônomos) |
| **EU AI Act (2024)** | Classifica AI systems for clinical decision support como alto risco — o que reforça a necessidade do humano no loop como requisito regulatório, não como limitação do projeto |
| **CFM Resolução 2.227/2018** | Sistemas de suporte à decisão clínica requerem documentação técnica e responsabilidade do médico pela decisão final |

O posicionamento como CDSS humano-no-loop é, portanto, a escolha regulatoriamente correta —
não uma limitação a ser justificada.

---

## 4. O que o MOSAIC-FL adiciona ao estado da arte como AI-based CDSS

A maioria dos papers de FL para saúde para no modelo: treinam, reportam AUC, e terminam.
O MOSAIC-FL fecha o ciclo do dado clínico à visualização do clínico.

| Contribuição | Por que é original |
|---|---|
| FL → ClinicalPath com distribuição completa de probabilidade + incerteza MC-Dropout como exames sintéticos | Papers de FL param na métrica — nenhum injeta a saída como exame na linha do tempo clínica |
| `correlation_token` efêmero para FHIR R4 | Resolve o campo obrigatório `subject` sem armazenar mapeamento identidade → token — design não documentado na literatura |
| FedProx + class weights por hospital | Combinação raramente explicitada em FL clínico — trata non-IID e desbalanceamento simultaneamente |
| CPU-aware com calibração explícita (temperature scaling + ECE) | FL clínico assume GPU; MOSAIC-FL é projetado para hardware hospitalar real |
| LGPD por construção (dados nunca saem + pseudonimização HMAC + audit trail) | A maioria dos papers de FL assume HIPAA americano; compliance LGPD em código é contribuição prática |

**O enquadramento CDSS fortalece essas contribuições** porque explica por que cada decisão
arquitetural existe:

- A distribuição completa de probabilidade não é capricho técnico — é o que permite ao clínico
  avaliar incerteza antes de decidir
- O FHIR R4 não é checkbox de interoperabilidade — é o que viabiliza integração com prontuários reais
- A LGPD não é compliance burocrático — é o que permite uso com dados reais de pacientes
- O RAG não gera diagnóstico autônomo — recupera casos similares como contexto para o clínico

---

## 5. O RAG no contexto do CDSS

O módulo RAG foi explicitamente projetado para o paradigma humano-no-loop:

- **O que o RAG faz:** recupera casos históricos similares (via MiniLM + similaridade de cosseno)
  e gera texto de contexto para apresentar ao clínico
- **O que o RAG não faz:** não emite parecer clínico autônomo; não substitui avaliação médica
- **Avaliação correta do RAG:** Precision@k dos casos recuperados (relevância dos casos para o
  desfecho real), não ROUGE ou BERTScore da geração textual
- **DistilGPT-2:** serve como scaffolding textual para apresentar os casos ao clínico — suas
  limitações de idioma (pré-treinado em inglês) são documentadas e a substituição por modelo
  multilíngue em português (Sabia-3, Llama-PT) consta do roadmap

### Ajuste de nomenclatura identificado

Os campos `confiavel` e `alucinacao_detectada` no retorno de `explain()` comunicam semântica
de sistema autônomo, contradizendo o design humano-no-loop. Proposta de renomeação:

```python
# Atual — semântica de sistema autônomo
return {
    "alucinacao_detectada": hallucinated,
    "confiavel": not hallucinated and len(sources) > 0,
}

# Proposto — semântica de CDSS
return {
    "requer_revisao_clinica": True,          # sempre, por design
    "casos_recuperados": len(sources) > 0,
}
```

---

## 6. Frase síntese para a defesa

> *"A contribuição não é um novo algoritmo de FL — é a primeira implementação documentada
> de um AI-based CDSS federado que fecha o ciclo do dado clínico à visualização do clínico,
> com incerteza quantificada, interoperabilidade FHIR R4 e compliance LGPD, projetado para
> hardware hospitalar sem GPU."*

---

## 7. Referências de base para o enquadramento

- **Topol, E.J. (2019).** High-performance medicine: the convergence of human and artificial
  intelligence. *Nature Medicine*, 25, 44–56.
  → Referência central para o paradigma "Augmented Intelligence": IA amplifica o clínico, não o substitui.

- **Rajpurkar, P. et al. (2022).** AI in health and medicine. *Nature Medicine*, 28, 31–38.
  → Estado da arte em AI-based CDSS; distingue sistemas autônomos de sistemas de apoio.

- **ANVISA. RDC 657/2022.** Regulamento técnico sobre Software como Dispositivo Médico (SaMD).

- **FDA. (2021).** Artificial Intelligence/Machine Learning (AI/ML)-Based Software as a
  Medical Device (SaMD) Action Plan.

- **CFM. Resolução 2.227/2018.** Define e disciplina o uso de sistemas de suporte à decisão
  clínica na prática médica.

---

*Documento gerado em 2026-06-24 a partir da discussão de enquadramento do MOSAIC-FL.*
