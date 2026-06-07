# LGPD — Roadmap de Conformidade para o MOSAIC-FL

Este documento cataloga os requisitos da Lei Geral de Proteção de Dados (Lei nº 13.709/2018) que têm impacto direto na engenharia de software do projeto. Cada item descreve o artigo, o que é exigido, o estado atual e o que precisaria ser implementado.

**Status atual:** este projeto executa com dados sintéticos em rede local. Nenhum dado real de paciente é processado agora. Os itens abaixo são pré-requisitos para que dados reais possam ser usados.

---

## Resumo executivo

| Item | Artigo LGPD | Impacto SE | Complexidade | Status |
|---|---|---|---|---|
| Trilha de auditoria | Art. 37 | Módulo novo (`audit_log.py`) | Média | Pendente |
| Differential Privacy nos pesos | Art. 46 | Mudança em `client_v2.py` | Alta | Pendente |
| Minimização de dados | Art. 6, III | Mudança em `preprocess_v2.py` | Baixa | Pendente |
| Pseudonimização de identificadores | Art. 13 §2 | Mudança em `preprocess_v2.py` | Baixa | Pendente |
| Controle de consentimento | Art. 7, I | Módulo novo + esquema de dados | Alta | Pendente |
| Retenção e exclusão de dados | Art. 15/16 | Scheduler task + esquema DB | Alta | Pendente |
| Controle de acesso / autenticação | Art. 46 | Novo middleware + RBAC | Alta | Pendente |
| Notificação de incidentes | Art. 48 | Alertas + integração ops | Média | Pendente |

---

## 1. Trilha de Auditoria (Art. 37)

**O que a lei exige:**
> "O controlador deve manter registro das operações de tratamento de dados pessoais que realizar."

**O que isso significa em código:**

O logging operacional atual (`logger.info("round_started", extra={...})`) serve para observabilidade. Para LGPD, cada acesso a dado pessoal precisa de um registro imutável e separado, com: quem acessou, qual dado, com qual finalidade, quando.

**O que precisaria ser implementado:**

```
infrastructure/audit_log.py          ← módulo de auditoria LGPD
```

Eventos que devem ser registrados:
- `training_access` — cliente carregou dados EHR para treinamento
- `weight_transmission` — pesos do modelo foram enviados ao servidor
- `model_export` — checkpoint foi salvo
- `data_rejected` — registro rejeitado por não ter consentimento
- `consent_check` — verificação de consentimento realizada

Estrutura de cada entrada no log de auditoria (JSONL):
```json
{
  "timestamp": "2026-06-07T10:30:00.000Z",
  "event_type": "training_access",
  "actor": "hospital_a",
  "purpose": "fl_training_round_5",
  "data_category": "ehr_pseudonymized",
  "record_count": 843,
  "round_num": 5,
  "legal_basis": "art_7_ix_pesquisa"
}
```

O log de auditoria deve ser:
- Separado do log operacional (arquivo `audit.jsonl` diferente do `server_daemon.log`)
- Não-rotacionável sem backup (integridade)
- Consultável por base jurídica e por período

---

## 2. Differential Privacy nos Pesos (Art. 46 — Segurança)

**O que a lei exige:**
> "Os agentes de tratamento devem adotar medidas de segurança, técnicas e administrativas aptas a proteger os dados pessoais de acessos não autorizados e de situações acidentais ou ilícitas de destruição, perda, alteração, comunicação ou qualquer forma de tratamento inadequado ou ilícito."

**Por que DP é relevante:**

Sem Differential Privacy, os pesos do modelo enviados ao servidor podem vazar informação sobre os dados de treinamento através de ataques conhecidos:
- **Model Inversion Attack** — reconstruir características individuais a partir dos pesos
- **Membership Inference Attack** — descobrir se um paciente específico estava no conjunto de treino

**O que precisaria ser implementado:**

Mecanismo Gaussiano de DP Local no `FedProxClient.get_parameters()`:

```python
# Em client_v2.py
def fit(self, parameters, config):
    self.set_parameters(parameters)
    self._initial_params = [p.copy() for p in parameters]  # guarda ponto inicial
    # ... treino ...
    return self.get_parameters(config), n_samples, metrics

def get_parameters(self, config):
    params = [v.cpu().numpy() for v in self.model.state_dict().values()]
    if self._dp_noise_multiplier > 0 and self._initial_params:
        # 1. Calcula delta (update) em vez de pesos absolutos
        deltas = [p - p0 for p, p0 in zip(params, self._initial_params)]
        # 2. Clipping — limita a sensibilidade global
        total_norm = sqrt(sum(||d||^2 for d in deltas))
        clip = min(1.0, self._dp_max_grad_norm / (total_norm + 1e-8))
        clipped = [d * clip for d in deltas]
        # 3. Ruído Gaussiano calibrado ao noise_multiplier
        noise_scale = self._dp_noise_multiplier * self._dp_max_grad_norm
        noised = [d + N(0, noise_scale, d.shape) for d in clipped]
        # 4. Retorna pesos_iniciais + delta_ruidoso
        return [p0 + d for p0, d in zip(self._initial_params, noised)]
    return params
```

**Parâmetros a configurar:**
```
FL_DP_NOISE_MULTIPLIER=0.0   # 0 = desabilitado; 0.5–1.0 = proteção razoável
FL_DP_MAX_GRAD_NORM=1.0      # norma de clipping
```

**Trade-off crítico:** valores maiores de `noise_multiplier` aumentam a proteção mas reduzem a acurácia do modelo. Recomendado calibrar empiricamente com os dados reais.

---

## 3. Minimização de Dados (Art. 6, III)

**O que a lei exige:**
> "A coleta é limitada ao mínimo necessário para a realização de suas finalidades, com abrangência dos dados pertinentes, proporcionais e não excessivos em relação às finalidades do tratamento de dados."

**O que isso significa em código:**

O `EHRPreprocessor` hoje processa todas as colunas que chegam no DataFrame. Em produção, colunas que não são necessárias para o modelo (histórico de pagamentos, dados socioeconômicos não pertinentes, endereço completo) não deveriam nem chegar ao `DataLoader`.

**O que precisaria ser implementado:**

```python
# Em preprocess_v2.py
ALLOWED_EHR_FIELDS: frozenset[str] = frozenset({
    "desfecho", "outcome", "faixa_etaria", "sexo", "genero",
    "sintoma", "exame", "diagnostico", "cid_codigo",
    "instituicao", "data_entrada", "data_saida",
    # colunas derivadas pelo preprocessor:
    "sintoma_encoded", "exame_encoded", "diagnostico_encoded",
    "idade", "peso",  # após normalização de unidades
})

def enforce_data_minimization(self, df: pd.DataFrame) -> pd.DataFrame:
    """Descarta colunas não necessárias para o treinamento. LGPD Art. 6 III."""
    non_essential = [c for c in df.columns if c.lower() not in ALLOWED_EHR_FIELDS]
    if non_essential:
        logger.warning("lgpd_minimization: descartando colunas %s", non_essential)
        df = df.drop(columns=non_essential)
    return df
```

**Nota:** `ALLOWED_EHR_FIELDS` deve ser auditável (revisado por DPO) e registrado no ROPA (Registro das Atividades de Processamento).

---

## 4. Pseudonimização de Identificadores Diretos (Art. 13 §2)

**O que a lei exige:**
> "Os dados tratados com base nesse parágrafo não poderão ser atribuídos ao indivíduo, salvo mediante o uso de informação adicional mantida separadamente pelo controlador em ambiente controlado e seguro."

**Identificadores diretos que NUNCA devem entrar no DataLoader:**

| Campo | Razão |
|---|---|
| `cpf` | Identificador único inequívoco |
| `nome`, `sobrenome` | Nome próprio |
| `prontuario`, `nr_prontuario` | ID interno do hospital |
| `data_nascimento` | Combinado com outros campos → re-identificação |
| `rg`, `cns`, `cartao_sus` | Documentos pessoais |
| `endereco`, `cep` | Localização |
| `email`, `telefone` | Contato direto |

**O que precisaria ser implementado:**

```python
DIRECT_IDENTIFIERS: frozenset[str] = frozenset({
    "cpf", "nome", "name", "sobrenome",
    "prontuario", "nr_prontuario", "patient_id",
    "rg", "cns", "cartao_sus",
    "data_nascimento", "birth_date",
    "endereco", "cep", "email", "telefone",
})

def pseudonymize_identifiers(self, df: pd.DataFrame) -> pd.DataFrame:
    """Substitui identificadores por hash SHA-256. LGPD Art. 13 §2."""
    cols = [c for c in df.columns if c.lower() in DIRECT_IDENTIFIERS]
    for col in cols:
        df[col] = df[col].astype(str).apply(
            lambda v: hashlib.sha256(v.encode()).hexdigest()[:16]
        )
        # registra no audit log que a pseudonimização ocorreu
    return df
```

**Importante:** o hash SHA-256 mantém a linkabilidade interna (o mesmo CPF sempre gera o mesmo hash) sem expor o valor original. Mas atenção — se o campo tem poucos valores únicos (ex: cidade pequena), o hash pode ser revertido por força bruta. Nesse caso, usar SHA-256 com salt secreto.

---

## 5. Controle de Consentimento (Art. 7, I)

**O que a lei exige:**
> "O tratamento de dados pessoais somente poderá ser realizado mediante o fornecimento de consentimento pelo titular."

**O que isso significa em código:**

Antes de um registro clínico entrar no DataLoader, o sistema precisa verificar que aquele paciente (identificado via hash do prontuário) consentiu com o uso dos dados para pesquisa em IA.

**O que precisaria ser implementado:**

```
infrastructure/consent/
    consent_store.py     ← consulta BD de consentimentos (SQLite/PostgreSQL)
    consent_checker.py   ← filtra DataFrame: mantém só registros com consentimento ativo
```

Interface mínima:
```python
class ConsentChecker:
    def filter_consented(self, df: pd.DataFrame, patient_id_col: str) -> pd.DataFrame:
        """Remove registros sem consentimento ativo. Loga no audit_log."""
        ...

    def has_consent(self, patient_hash: str) -> bool:
        """Consulta banco de consentimentos por hash do paciente."""
        ...
```

**Variável de controle:**
```
FL_REQUIRE_CONSENT=true   # se true, bloqueia treinamento sem arquivo de consentimentos
```

**Nota prática:** para dados de pesquisa com aprovação de CEP (Comitê de Ética em Pesquisa), o Art. 7, IX permite uso sem consentimento individual. É necessário documentar a base legal usada e registrar no ROPA.

---

## 6. Retenção e Exclusão de Dados (Art. 15 e 16)

**O que a lei exige:**
> "O término do tratamento de dados pessoais ocorrerá quando verificada a finalidade ou o período especificado para o término do tratamento."

**O que isso significa em código:**

1. Os pesos do modelo (checkpoints) contêm informação derivada dos dados dos pacientes. Precisam de política de retenção.
2. Os logs que contenham dados pessoais (mesmo pseudonimizados) precisam de expiração.
3. Se um paciente exercer o direito de exclusão (Art. 18, VI), os dados derivados daquele paciente precisam ser identificados e removidos.

**O que precisaria ser implementado:**

```python
# No scheduler: tarefa periódica de limpeza
class DataRetentionTask:
    def purge_expired_checkpoints(self, max_age_days: int = 365) -> None: ...
    def purge_expired_audit_logs(self, max_age_days: int = 1825) -> None:  # 5 anos
        # LGPD exige guarda por prazo razoável para prestação de contas
        ...
    def process_deletion_requests(self, requests: list[str]) -> None:
        # Para cada patient_hash em requests: remove da base local
        # Não remove pesos do modelo (tecnicamente não é possível sem re-treino)
        # Documenta no audit_log que o pedido foi atendido
        ...
```

**Problema não resolvível sem re-treino:** os pesos do modelo global são uma função não-invertível de todos os dados de treinamento. Se um paciente pede exclusão, tecnicamente é necessário re-treinar o modelo excluindo aquele paciente — o que é computacionalmente caro. A abordagem prática é documentar essa limitação no ROPA e na política de privacidade, e comprometer-se com re-treino a cada N períodos.

---

## 7. Controle de Acesso e Autenticação (Art. 46)

**O que a lei exige:**
> "Os agentes de tratamento devem adotar medidas de segurança, técnicas e administrativas aptas a proteger os dados pessoais."

**Estado atual:**
Não há autenticação entre servidor e cliente Flower. Qualquer processo que conheça o endereço e porta pode conectar como cliente.

**O que precisaria ser implementado:**

Com a implementação de TLS (já feita), o canal é criptografado. Para autenticação do cliente, o Flower 1.x suporta autenticação via chaves EC (`authentication_keys` no `start_client`). Alternativamente, mutual TLS (mTLS) com certificados de cliente — cada hospital recebe um certificado assinado pela CA do projeto.

```bash
# mTLS: gerar certificado por hospital
openssl req -newkey rsa:4096 -keyout hospital_a.key -out hospital_a.csr -nodes
openssl x509 -req -in hospital_a.csr -CA ca.crt -CAkey ca.key -out hospital_a.crt -days 365
```

No servidor, verificar que o certificado apresentado pertence a um hospital autorizado (whitelist de CN/fingerprint).

---

## 8. Notificação de Incidentes (Art. 48)

**O que a lei exige:**
> "O controlador deverá comunicar à autoridade nacional e ao titular a ocorrência de incidente de segurança que possa acarretar risco ou dano relevante, em prazo razoável."

**O que isso significa em código:**

Detectar e alertar automaticamente sobre eventos suspeitos:
- Tentativas de conexão de cliente não autorizado
- Acesso anômalo ao banco de consentimentos
- Exfiltração de checkpoints (cópia não autorizada)
- Falhas de autenticação repetidas (brute force)

**O que precisaria ser implementado:**

Um detector de anomalias simples no log de auditoria, com alertas via email/webhook para o DPO (Encarregado de Proteção de Dados). O prazo regulatório é "o mais breve possível" — a ANPD considera 72h como referência.

---

## Ordem de implementação recomendada

Considerando impacto de proteção vs. esforço de desenvolvimento:

1. **Pseudonimização** (item 4) — baixa complexidade, alto impacto. Bloqueador para dados reais.
2. **Minimização de dados** (item 3) — baixa complexidade. Configurar `ALLOWED_EHR_FIELDS`.
3. **Trilha de auditoria** (item 1) — média complexidade. Necessária para demonstrar conformidade.
4. **Differential Privacy** (item 2) — alta complexidade. Requer calibração experimental.
5. **Consentimento** (item 5) — alta complexidade. Requer integração com sistema hospitalar.
6. **Retenção e exclusão** (item 6) — alta complexidade. Scheduler task + política documentada.
7. **mTLS / RBAC** (item 7) — alta complexidade. Depende de PKI do hospital.
8. **Notificação de incidentes** (item 8) — média complexidade. Depende de ops/infra.

---

## Base legal para pesquisa (Art. 7, IX e Art. 13, §3)

Para projetos de pesquisa científica, a LGPD permite uso de dados sem consentimento individual quando:
- Há aprovação de **CEP** (Comitê de Ética em Pesquisa)
- Os dados são tratados de forma **pseudonimizada**
- O uso é limitado à finalidade da pesquisa aprovada
- O ROPA documenta a base legal, a finalidade e as medidas de segurança

Isso reduz (mas não elimina) os itens 1, 3, 4 acima como pré-requisitos para uso de dados reais com aprovação ética vigente.

---

*Documento gerado em 2026-06-07. Revisar com advogado especializado em LGPD e com o DPO da instituição antes de processar dados reais de pacientes.*
