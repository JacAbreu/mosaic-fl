-- migration 010: rastreabilidade de treinamentos FL
--
-- Cada execução de run_training.py registra 1 linha em fl_trainings antes de
-- iniciar o loop federado. O checkpoint guloso faz UPSERT nessa linha (1 checkpoint
-- por treinamento). load_best() filtra por training_id — sem cross-contamination
-- entre execuções.

-- Tabela de treinamentos
CREATE TABLE IF NOT EXISTS metrics.fl_trainings (
    id              SERIAL PRIMARY KEY,
    algorithm       TEXT        NOT NULL DEFAULT 'FedAvg',
    log_file        TEXT        NOT NULL DEFAULT '',
    n_rounds_max    INTEGER     NOT NULL DEFAULT 120,
    started_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at    TIMESTAMPTZ,
    status          TEXT        NOT NULL DEFAULT 'running',   -- running | completed | failed
    n_rounds_done   INTEGER,
    best_round      INTEGER,
    best_accuracy   REAL,
    converged       BOOLEAN
);

-- Adiciona training_id em fl_checkpoints (nullable para compatibilidade com dados anteriores)
ALTER TABLE metrics.fl_checkpoints
    ADD COLUMN IF NOT EXISTS training_id INTEGER REFERENCES metrics.fl_trainings(id);

-- Garante 1 checkpoint por treinamento (UPSERT usa este índice)
CREATE UNIQUE INDEX IF NOT EXISTS fl_checkpoints_training_id_uniq
    ON metrics.fl_checkpoints (training_id)
    WHERE training_id IS NOT NULL;
