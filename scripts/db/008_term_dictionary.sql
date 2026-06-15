-- 008_term_dictionary.sql
-- Cria knowledge.term_dictionary — dicionário persistente de termos semânticos.
--
-- Centraliza dois tipos de mapeamento que antes viviam em código estático:
--   1. 'column_concept' — aliases de colunas CSV → conceito semântico
--      (ex: 'de_analito', 'analito', 'nm_analito' → 'analyte')
--   2. 'analyte' — aliases de nomes de analitos → nome canônico
--      (ex: 'WBC', 'leucocitos', 'Leucócitos' → 'LEUCOCITOS')
--
-- O campo `source` identifica a origem de cada alias (FAPESP, CLINICAL_PATH,
-- SBPC, MANUAL), permitindo auditoria de quais termos chegaram de cada sistema
-- externo e rastreabilidade dos pesos do modelo federated learning.
--
-- Extensão: novos sistemas externos adicionam apenas linhas — sem alteração de código.

CREATE TABLE IF NOT EXISTS knowledge.term_dictionary (
    id          BIGSERIAL PRIMARY KEY,
    term_type   TEXT        NOT NULL,
    canonical   TEXT        NOT NULL,
    alias       TEXT        NOT NULL,
    source      TEXT        NOT NULL DEFAULT 'MANUAL',
    active      BOOLEAN     NOT NULL DEFAULT TRUE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT term_dictionary_unique UNIQUE (term_type, canonical, alias)
);

COMMENT ON TABLE  knowledge.term_dictionary            IS 'Persistent semantic term dictionary. Maps aliases to canonical concepts for column resolution (term_type=column_concept) and analyte normalization (term_type=analyte). Source of truth for ColumnResolver and canonical_refs module.';
COMMENT ON COLUMN knowledge.term_dictionary.term_type  IS 'Discriminator: column_concept | analyte | outcome_class (extensible).';
COMMENT ON COLUMN knowledge.term_dictionary.canonical  IS 'Canonical form. For column_concept: semantic concept name (patient_id, analyte). For analyte: uppercase normalized name (LEUCOCITOS, HEMOGLOBINA).';
COMMENT ON COLUMN knowledge.term_dictionary.alias      IS 'Alternative name that resolves to canonical. Stored as received from source; normalization applied at query time.';
COMMENT ON COLUMN knowledge.term_dictionary.source     IS 'Origin of this alias: FAPESP | CLINICAL_PATH | SBPC | MANUAL.';
COMMENT ON COLUMN knowledge.term_dictionary.active     IS 'FALSE = soft-deleted; excluded from resolution but preserved for audit.';

CREATE INDEX IF NOT EXISTS term_dictionary_lookup_idx
    ON knowledge.term_dictionary (term_type, active)
    INCLUDE (canonical, alias);

CREATE INDEX IF NOT EXISTS term_dictionary_canonical_idx
    ON knowledge.term_dictionary (term_type, canonical)
    WHERE active = TRUE;
