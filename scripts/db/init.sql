-- init.sql
-- Executado automaticamente na primeira inicialização do container.
-- Cria extensões, schemas e tabelas estáticas.
-- Tabelas gerenciadas pela aplicação (SQLAlchemy) são criadas em runtime.

-- ---------------------------------------------------------------------------
-- Extensões
-- ---------------------------------------------------------------------------
CREATE EXTENSION IF NOT EXISTS timescaledb;
CREATE EXTENSION IF NOT EXISTS vector;

-- ---------------------------------------------------------------------------
-- Schemas
-- ---------------------------------------------------------------------------
CREATE SCHEMA IF NOT EXISTS clinical;   -- cadastro de pacientes + config FL (PostgreSQL puro)
CREATE SCHEMA IF NOT EXISTS metrics;    -- séries temporais de exames e risco (TimescaleDB)
CREATE SCHEMA IF NOT EXISTS knowledge;  -- perfis clínicos prototípicos para RAG (pgvector)

-- ---------------------------------------------------------------------------
-- Permissões
-- ---------------------------------------------------------------------------
GRANT ALL ON SCHEMA clinical  TO mosaicfl;
GRANT ALL ON SCHEMA metrics   TO mosaicfl;
GRANT ALL ON SCHEMA knowledge TO mosaicfl;

-- ---------------------------------------------------------------------------
-- Schema: clinical
-- Tabelas estáticas — sem dimensão temporal relevante.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS clinical.patients (
    patient_id  TEXT PRIMARY KEY,
    sex         TEXT NOT NULL DEFAULT 'M',
    age         REAL NOT NULL DEFAULT 0.0
);

CREATE TABLE IF NOT EXISTS clinical.export_paths (
    patient_id   TEXT PRIMARY KEY,
    export_path  TEXT NOT NULL
);

-- Config de runtime lida pelo servidor FL antes de cada round.
-- Uma única linha (id='current') sobrescrita pelo operador.
CREATE TABLE IF NOT EXISTS clinical.fl_orchestration_config (
    id              TEXT PRIMARY KEY DEFAULT 'current',
    proximal_mu     REAL,               -- NULL = usa o valor padrão da strategy
    pause_seconds   REAL NOT NULL DEFAULT 0.0,
    stop            BOOLEAN NOT NULL DEFAULT FALSE,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Garante que sempre existe uma linha com defaults
INSERT INTO clinical.fl_orchestration_config (id)
VALUES ('current')
ON CONFLICT DO NOTHING;

-- ---------------------------------------------------------------------------
-- Schema: metrics
-- Séries temporais — hypertables TimescaleDB particionadas por date.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS metrics.risk_history (
    patient_id  TEXT NOT NULL,
    date        DATE NOT NULL,
    risk_score  REAL NOT NULL
);
SELECT create_hypertable(
    'metrics.risk_history', 'date',
    if_not_exists => TRUE
);

CREATE TABLE IF NOT EXISTS metrics.exam_records (
    patient_id    TEXT NOT NULL,
    exam_name     TEXT NOT NULL,
    date          DATE NOT NULL,
    value         REAL NOT NULL,
    phase         TEXT NOT NULL,
    ref_low       REAL NOT NULL DEFAULT 0.0,
    ref_high      REAL NOT NULL DEFAULT 0.0,
    sex_ref_low   REAL NOT NULL DEFAULT 0.0,
    sex_ref_high  REAL NOT NULL DEFAULT 0.0
);
SELECT create_hypertable(
    'metrics.exam_records', 'date',
    if_not_exists => TRUE
);

-- ---------------------------------------------------------------------------
-- Schema: knowledge
-- Perfis clínicos prototípicos anonimizados para recuperação por similaridade.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS knowledge.clinical_profiles (
    id          TEXT PRIMARY KEY,                    -- "profile_0", "profile_1", ...
    document    TEXT NOT NULL,                       -- texto do perfil anonimizado
    embedding   VECTOR(384) NOT NULL,               -- all-MiniLM-L6-v2 (384 dims)
    desfecho    TEXT,
    faixa_etaria TEXT,
    categoria   TEXT
);

-- Índice HNSW para busca por similaridade de cosseno (melhor para top-k clínico)
CREATE INDEX IF NOT EXISTS clinical_profiles_embedding_idx
    ON knowledge.clinical_profiles
    USING hnsw (embedding vector_cosine_ops);
