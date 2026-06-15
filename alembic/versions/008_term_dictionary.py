"""term_dictionary

Cria knowledge.term_dictionary — dicionário persistente de termos semânticos.

Centraliza dois tipos de mapeamento que antes viviam em código estático:
  1. 'column_concept' — aliases de colunas CSV → conceito semântico
     (ex: 'de_analito', 'analito', 'nm_analito' → 'analyte')
     Antes: CLINICAL_SEMANTIC_MAP em integration/column_resolver.py
  2. 'analyte' — aliases de nomes de analitos → nome canônico
     (ex: 'WBC', 'leucocitos', 'Leucócitos' → 'LEUCOCITOS')
     Base inicial: EXAM_CATALOG em wire-production/seed/generate_data.py

O campo `source` identifica a origem de cada alias ('FAPESP', 'CLINICAL_PATH',
'SBPC', 'MANUAL'), permitindo auditoria de quais termos chegaram de cada
sistema externo e rastreabilidade dos pesos do modelo federated learning.

Revision ID: 008
Revises: 007
Create Date: 2026-06-14
"""
from typing import Sequence, Union

from alembic import op


revision: str = '008'
down_revision: Union[str, Sequence[str], None] = '007'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
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

        -- ----------------------------------------------------------------
        -- Seed: column_concept — CLINICAL_SEMANTIC_MAP
        -- ----------------------------------------------------------------
        INSERT INTO knowledge.term_dictionary (term_type, canonical, alias, source) VALUES

        -- patients
        ('column_concept', 'patient_id',   'id_paciente',          'FAPESP'),
        ('column_concept', 'patient_id',   'id_patient',           'MANUAL'),
        ('column_concept', 'patient_id',   'patientid',            'MANUAL'),
        ('column_concept', 'patient_id',   'cod_paciente',         'FAPESP'),
        ('column_concept', 'patient_id',   'paciente_id',          'MANUAL'),

        ('column_concept', 'sex',          'ic_sexo',              'FAPESP'),
        ('column_concept', 'sex',          'sexo',                 'FAPESP'),
        ('column_concept', 'sex',          'sex',                  'MANUAL'),
        ('column_concept', 'sex',          'gender',               'MANUAL'),
        ('column_concept', 'sex',          'genero',               'MANUAL'),
        ('column_concept', 'sex',          'sexo_paciente',        'FAPESP'),

        ('column_concept', 'birth_year',   'aa_nascimento',        'FAPESP'),
        ('column_concept', 'birth_year',   'ano_nascimento',       'FAPESP'),
        ('column_concept', 'birth_year',   'birth_year',           'MANUAL'),
        ('column_concept', 'birth_year',   'anascimento',          'FAPESP'),
        ('column_concept', 'birth_year',   'aa_nasc',              'FAPESP'),

        ('column_concept', 'state_code',   'cd_uf',                'FAPESP'),
        ('column_concept', 'state_code',   'uf',                   'FAPESP'),
        ('column_concept', 'state_code',   'estado',               'FAPESP'),
        ('column_concept', 'state_code',   'state',                'MANUAL'),
        ('column_concept', 'state_code',   'cd_estado',            'FAPESP'),
        ('column_concept', 'state_code',   'sg_uf',                'FAPESP'),

        ('column_concept', 'municipality', 'cd_municipio',         'FAPESP'),
        ('column_concept', 'municipality', 'municipio',            'FAPESP'),
        ('column_concept', 'municipality', 'municipality',         'MANUAL'),
        ('column_concept', 'municipality', 'nm_municipio',         'FAPESP'),
        ('column_concept', 'municipality', 'cd_mun',               'FAPESP'),

        ('column_concept', 'cep_prefix',   'cd_cepreduzido',       'FAPESP'),
        ('column_concept', 'cep_prefix',   'cep_reduzido',         'FAPESP'),
        ('column_concept', 'cep_prefix',   'cep_prefix',           'MANUAL'),
        ('column_concept', 'cep_prefix',   'cd_cep',               'FAPESP'),
        ('column_concept', 'cep_prefix',   'cepreduzido',          'FAPESP'),

        ('column_concept', 'hospital_id',  'de_hospital',          'FAPESP'),
        ('column_concept', 'hospital_id',  'hospital',             'FAPESP'),
        ('column_concept', 'hospital_id',  'institution',          'MANUAL'),
        ('column_concept', 'hospital_id',  'hospital_id',          'MANUAL'),
        ('column_concept', 'hospital_id',  'nm_hospital',          'FAPESP'),

        -- attendances
        ('column_concept', 'attendance_id',   'id_atendimento',       'FAPESP'),
        ('column_concept', 'attendance_id',   'atendimento_id',       'FAPESP'),
        ('column_concept', 'attendance_id',   'attendance_id',        'MANUAL'),
        ('column_concept', 'attendance_id',   'nr_atendimento',       'FAPESP'),

        ('column_concept', 'attended_at',     'dt_atendimento',       'FAPESP'),
        ('column_concept', 'attended_at',     'data_atendimento',     'FAPESP'),
        ('column_concept', 'attended_at',     'attended_at',          'MANUAL'),
        ('column_concept', 'attended_at',     'dt_internacao',        'FAPESP'),

        ('column_concept', 'attendance_type', 'de_tipo_atendimento',  'FAPESP'),
        ('column_concept', 'attendance_type', 'tipo_atendimento',     'FAPESP'),
        ('column_concept', 'attendance_type', 'tp_atendimento',       'FAPESP'),
        ('column_concept', 'attendance_type', 'modality',             'MANUAL'),

        ('column_concept', 'specialty',       'de_clinica',           'FAPESP'),
        ('column_concept', 'specialty',       'clinica',              'FAPESP'),
        ('column_concept', 'specialty',       'specialty',            'MANUAL'),
        ('column_concept', 'specialty',       'especialidade',        'FAPESP'),
        ('column_concept', 'specialty',       'nm_clinica',           'FAPESP'),

        ('column_concept', 'clinic_id',       'id_clinica',           'FAPESP'),
        ('column_concept', 'clinic_id',       'clinic_id',            'MANUAL'),
        ('column_concept', 'clinic_id',       'id_clinic',            'MANUAL'),
        ('column_concept', 'clinic_id',       'nr_clinica',           'FAPESP'),
        ('column_concept', 'clinic_id',       'cd_clinica',           'FAPESP'),

        ('column_concept', 'suspected_diagnosis', 'suspected_diagnosis',    'MANUAL'),
        ('column_concept', 'suspected_diagnosis', 'hipotese_diagnostica',   'FAPESP'),
        ('column_concept', 'suspected_diagnosis', 'hipotese_diagnostico',   'FAPESP'),
        ('column_concept', 'suspected_diagnosis', 'ds_hipotese',            'FAPESP'),
        ('column_concept', 'suspected_diagnosis', 'cd_hipotese',            'FAPESP'),
        ('column_concept', 'suspected_diagnosis', 'diagnostico_provavel',   'FAPESP'),
        ('column_concept', 'suspected_diagnosis', 'probable_diagnosis',     'MANUAL'),
        ('column_concept', 'suspected_diagnosis', 'working_diagnosis',      'MANUAL'),
        ('column_concept', 'suspected_diagnosis', 'cid_suspeito',           'FAPESP'),
        ('column_concept', 'suspected_diagnosis', 'cid_hipotese',           'FAPESP'),
        ('column_concept', 'suspected_diagnosis', 'cd_cid_hipotese',        'FAPESP'),

        ('column_concept', 'confirmed_diagnosis', 'confirmed_diagnosis',    'MANUAL'),
        ('column_concept', 'confirmed_diagnosis', 'diagnostico_confirmado', 'FAPESP'),
        ('column_concept', 'confirmed_diagnosis', 'diagnostico_definitivo', 'FAPESP'),
        ('column_concept', 'confirmed_diagnosis', 'ds_diagnostico',         'FAPESP'),
        ('column_concept', 'confirmed_diagnosis', 'cd_diagnostico',         'FAPESP'),
        ('column_concept', 'confirmed_diagnosis', 'cid_confirmado',         'FAPESP'),
        ('column_concept', 'confirmed_diagnosis', 'definitive_diagnosis',   'MANUAL'),
        ('column_concept', 'confirmed_diagnosis', 'final_diagnosis',        'MANUAL'),
        ('column_concept', 'confirmed_diagnosis', 'cid_principal',          'FAPESP'),
        ('column_concept', 'confirmed_diagnosis', 'cd_cid_definitivo',      'FAPESP'),
        ('column_concept', 'confirmed_diagnosis', 'cd_cid_principal',       'FAPESP'),

        -- exam records
        ('column_concept', 'collection_date', 'dt_coleta',             'FAPESP'),
        ('column_concept', 'collection_date', 'data_coleta',           'FAPESP'),
        ('column_concept', 'collection_date', 'collection_date',       'MANUAL'),
        ('column_concept', 'collection_date', 'dt_exame',              'FAPESP'),

        ('column_concept', 'origin',          'de_origem',             'FAPESP'),
        ('column_concept', 'origin',          'origem',                'FAPESP'),
        ('column_concept', 'origin',          'origin',                'MANUAL'),
        ('column_concept', 'origin',          'source',                'MANUAL'),
        ('column_concept', 'origin',          'local_coleta',          'FAPESP'),

        ('column_concept', 'exam_group',      'de_exame',              'FAPESP'),
        ('column_concept', 'exam_group',      'exame',                 'FAPESP'),
        ('column_concept', 'exam_group',      'exam_name',             'MANUAL'),
        ('column_concept', 'exam_group',      'grupo_exame',           'FAPESP'),
        ('column_concept', 'exam_group',      'nm_exame',              'FAPESP'),

        ('column_concept', 'analyte',         'de_analito',            'FAPESP'),
        ('column_concept', 'analyte',         'analito',               'FAPESP'),
        ('column_concept', 'analyte',         'analyte',               'MANUAL'),
        ('column_concept', 'analyte',         'nm_analito',            'FAPESP'),
        ('column_concept', 'analyte',         'analise',               'FAPESP'),

        ('column_concept', 'result_text',     'de_resultado',          'FAPESP'),
        ('column_concept', 'result_text',     'resultado',             'FAPESP'),
        ('column_concept', 'result_text',     'result',                'MANUAL'),
        ('column_concept', 'result_text',     'ds_resultado',          'FAPESP'),

        ('column_concept', 'result_num',      'de_resultnum',          'FAPESP'),
        ('column_concept', 'result_num',      'resultnum',             'FAPESP'),
        ('column_concept', 'result_num',      'valor_numerico',        'FAPESP'),
        ('column_concept', 'result_num',      'vl_resultado',          'FAPESP'),

        ('column_concept', 'unit',            'cd_unidade',            'FAPESP'),
        ('column_concept', 'unit',            'unidade',               'FAPESP'),
        ('column_concept', 'unit',            'unit',                  'MANUAL'),
        ('column_concept', 'unit',            'ds_unidade',            'FAPESP'),

        ('column_concept', 'reference_range', 'de_valor_referencia',   'FAPESP'),
        ('column_concept', 'reference_range', 'valor_referencia',      'FAPESP'),
        ('column_concept', 'reference_range', 'referencia',            'FAPESP'),
        ('column_concept', 'reference_range', 'vl_referencia',         'FAPESP'),

        -- outcomes
        ('column_concept', 'outcome_text',    'de_desfecho',           'FAPESP'),
        ('column_concept', 'outcome_text',    'desfecho',              'FAPESP'),
        ('column_concept', 'outcome_text',    'outcome',               'MANUAL'),
        ('column_concept', 'outcome_text',    'evolucao',              'FAPESP'),
        ('column_concept', 'outcome_text',    'ds_desfecho',           'FAPESP'),

        ('column_concept', 'outcome_date',    'dt_desfecho',           'FAPESP'),
        ('column_concept', 'outcome_date',    'data_desfecho',         'FAPESP'),
        ('column_concept', 'outcome_date',    'outcome_date',          'MANUAL'),
        ('column_concept', 'outcome_date',    'dt_alta',               'FAPESP'),

        ('column_concept', 'outcome_type',    'de_tipo_atendimento',   'FAPESP'),
        ('column_concept', 'outcome_type',    'tipo_atendimento',      'FAPESP'),
        ('column_concept', 'outcome_type',    'tp_atendimento',        'FAPESP'),

        -- ----------------------------------------------------------------
        -- Seed: analyte — nomes canônicos e aliases por sistema externo
        -- ----------------------------------------------------------------

        -- HEMOGLOBINA
        ('analyte', 'HEMOGLOBINA', 'HEMOGLOBINA',        'FAPESP'),
        ('analyte', 'HEMOGLOBINA', 'hemoglobina',        'FAPESP'),
        ('analyte', 'HEMOGLOBINA', 'HGB',                'CLINICAL_PATH'),
        ('analyte', 'HEMOGLOBINA', 'HB',                 'CLINICAL_PATH'),
        ('analyte', 'HEMOGLOBINA', 'HAEMOGLOBIN',        'MANUAL'),

        -- LEUCOCITOS
        ('analyte', 'LEUCOCITOS',  'LEUCOCITOS',         'FAPESP'),
        ('analyte', 'LEUCOCITOS',  'leucocitos',         'FAPESP'),
        ('analyte', 'LEUCOCITOS',  'Leucócitos',         'FAPESP'),
        ('analyte', 'LEUCOCITOS',  'WBC',                'CLINICAL_PATH'),
        ('analyte', 'LEUCOCITOS',  'WHITE BLOOD CELLS',  'MANUAL'),
        ('analyte', 'LEUCOCITOS',  'LEUKOCYTES',         'MANUAL'),

        -- PLAQUETAS
        ('analyte', 'PLAQUETAS',   'PLAQUETAS',          'FAPESP'),
        ('analyte', 'PLAQUETAS',   'plaquetas',          'FAPESP'),
        ('analyte', 'PLAQUETAS',   'PLT',                'CLINICAL_PATH'),
        ('analyte', 'PLAQUETAS',   'PLATELETS',          'MANUAL'),
        ('analyte', 'PLAQUETAS',   'THROMBOCYTES',       'MANUAL'),

        -- CREATININA
        ('analyte', 'CREATININA',  'CREATININA',         'FAPESP'),
        ('analyte', 'CREATININA',  'creatinina',         'FAPESP'),
        ('analyte', 'CREATININA',  'CREA',               'CLINICAL_PATH'),
        ('analyte', 'CREATININA',  'CR',                 'CLINICAL_PATH'),
        ('analyte', 'CREATININA',  'CREATININE',         'MANUAL'),

        -- UREIA
        ('analyte', 'UREIA',       'UREIA',              'FAPESP'),
        ('analyte', 'UREIA',       'ureia',              'FAPESP'),
        ('analyte', 'UREIA',       'BUN',                'CLINICAL_PATH'),
        ('analyte', 'UREIA',       'UREA',               'MANUAL'),
        ('analyte', 'UREIA',       'BLOOD UREA NITROGEN','MANUAL'),

        -- PCR
        ('analyte', 'PCR',         'PCR',                'FAPESP'),
        ('analyte', 'PCR',         'pcr',                'FAPESP'),
        ('analyte', 'PCR',         'CRP',                'CLINICAL_PATH'),
        ('analyte', 'PCR',         'C-REACTIVE PROTEIN', 'MANUAL'),
        ('analyte', 'PCR',         'PROTEINA C REATIVA', 'FAPESP'),

        -- FERRITINA
        ('analyte', 'FERRITINA',   'FERRITINA',          'FAPESP'),
        ('analyte', 'FERRITINA',   'ferritina',          'FAPESP'),
        ('analyte', 'FERRITINA',   'FER',                'CLINICAL_PATH'),
        ('analyte', 'FERRITINA',   'FERRITIN',           'MANUAL'),

        -- D_DIMERO
        ('analyte', 'D_DIMERO',    'D_DIMERO',           'FAPESP'),
        ('analyte', 'D_DIMERO',    'd_dimero',           'FAPESP'),
        ('analyte', 'D_DIMERO',    'D-DIMERO',           'FAPESP'),
        ('analyte', 'D_DIMERO',    'DDIMER',             'CLINICAL_PATH'),
        ('analyte', 'D_DIMERO',    'D-DIMER',            'CLINICAL_PATH'),
        ('analyte', 'D_DIMERO',    'D DIMER',            'MANUAL'),

        -- LACTATO
        ('analyte', 'LACTATO',     'LACTATO',            'FAPESP'),
        ('analyte', 'LACTATO',     'lactato',            'FAPESP'),
        ('analyte', 'LACTATO',     'LAC',                'CLINICAL_PATH'),
        ('analyte', 'LACTATO',     'LACTATE',            'MANUAL'),
        ('analyte', 'LACTATO',     'LACTIC ACID',        'MANUAL'),

        -- TGO
        ('analyte', 'TGO',         'TGO',                'FAPESP'),
        ('analyte', 'TGO',         'tgo',                'FAPESP'),
        ('analyte', 'TGO',         'AST',                'CLINICAL_PATH'),
        ('analyte', 'TGO',         'ASPARTATO AMINOTRANSFERASE', 'FAPESP'),
        ('analyte', 'TGO',         'ASPARTATE AMINOTRANSFERASE', 'MANUAL'),

        -- TGP
        ('analyte', 'TGP',         'TGP',                'FAPESP'),
        ('analyte', 'TGP',         'tgp',                'FAPESP'),
        ('analyte', 'TGP',         'ALT',                'CLINICAL_PATH'),
        ('analyte', 'TGP',         'ALANINA AMINOTRANSFERASE',  'FAPESP'),
        ('analyte', 'TGP',         'ALANINE AMINOTRANSFERASE',  'MANUAL'),

        -- BILIRRUBINA_TOT
        ('analyte', 'BILIRRUBINA_TOT', 'BILIRRUBINA_TOT',     'FAPESP'),
        ('analyte', 'BILIRRUBINA_TOT', 'bilirrubina_tot',     'FAPESP'),
        ('analyte', 'BILIRRUBINA_TOT', 'BILIRRUBINA TOTAL',   'FAPESP'),
        ('analyte', 'BILIRRUBINA_TOT', 'TBIL',                'CLINICAL_PATH'),
        ('analyte', 'BILIRRUBINA_TOT', 'TOTAL BILIRUBIN',     'MANUAL'),

        -- SODIO
        ('analyte', 'SODIO',       'SODIO',              'FAPESP'),
        ('analyte', 'SODIO',       'sodio',              'FAPESP'),
        ('analyte', 'SODIO',       'SÓDIO',              'FAPESP'),
        ('analyte', 'SODIO',       'NA',                 'CLINICAL_PATH'),
        ('analyte', 'SODIO',       'SODIUM',             'MANUAL'),

        -- POTASSIO
        ('analyte', 'POTASSIO',    'POTASSIO',           'FAPESP'),
        ('analyte', 'POTASSIO',    'potassio',           'FAPESP'),
        ('analyte', 'POTASSIO',    'POTÁSSIO',           'FAPESP'),
        ('analyte', 'POTASSIO',    'K',                  'CLINICAL_PATH'),
        ('analyte', 'POTASSIO',    'POTASSIUM',          'MANUAL'),

        -- GLICOSE
        ('analyte', 'GLICOSE',     'GLICOSE',            'FAPESP'),
        ('analyte', 'GLICOSE',     'glicose',            'FAPESP'),
        ('analyte', 'GLICOSE',     'GLUCOSE',            'CLINICAL_PATH'),
        ('analyte', 'GLICOSE',     'GLU',                'CLINICAL_PATH'),
        ('analyte', 'GLICOSE',     'BLOOD GLUCOSE',      'MANUAL')

        ON CONFLICT (term_type, canonical, alias) DO NOTHING;
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS knowledge.term_dictionary;")
