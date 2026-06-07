"""
seed/generate_data.py
Gera arquivos JSON de exames clínicos sintéticos para testar o watcher da API.

Cada arquivo pode ser copiado para o diretório "incoming" monitorado pela API,
que vai ingeri-lo, gerar o score de risco e exportar os arquivos ClinicalPath.

Uso:
    # Gera os JSONs em ./incoming/ (para teste com docker compose)
    python generate_data.py --output-dir ../incoming

    # Gera 5 pacientes com 10 exames cada
    python generate_data.py --patients 5 --exams-per-patient 10

    # Envia diretamente para a API (requer httpx: pip install httpx)
    python generate_data.py --send --api-url http://localhost:8000

Formato de cada arquivo:
    {
        "patient_id": "P001",
        "sex": "F",
        "age": 72.0,
        "exams": [
            {
                "exam_name": "HEMOGLOBINA",
                "date": "2024-01-15",
                "value": 11.2,
                "phase": "IN",
                "ref_low": 12.0,
                "ref_high": 17.5,
                "sex_ref_low": 11.5,
                "sex_ref_high": 15.5
            },
            ...
        ]
    }
"""
import argparse
import json
import random
import sys
from datetime import date, timedelta
from pathlib import Path

# Exames clínicos reais presentes no dataset COVID-19 do ClinicalPath
EXAM_CATALOG = [
    {"name": "HEMOGLOBINA",     "ref_low": 12.0,  "ref_high": 17.5,  "sex_f_low": 11.5, "sex_f_high": 15.5,  "unit": "g/dL"},
    {"name": "LEUCOCITOS",      "ref_low": 4000,  "ref_high": 11000, "sex_f_low": 4000,  "sex_f_high": 11000, "unit": "/mm3"},
    {"name": "PLAQUETAS",       "ref_low": 150000,"ref_high": 400000,"sex_f_low": 150000,"sex_f_high": 400000,"unit": "/mm3"},
    {"name": "CREATININA",      "ref_low": 0.6,   "ref_high": 1.2,   "sex_f_low": 0.5,   "sex_f_high": 1.1,   "unit": "mg/dL"},
    {"name": "UREIA",           "ref_low": 15.0,  "ref_high": 45.0,  "sex_f_low": 15.0,  "sex_f_high": 45.0,  "unit": "mg/dL"},
    {"name": "PCR",             "ref_low": 0.0,   "ref_high": 5.0,   "sex_f_low": 0.0,   "sex_f_high": 5.0,   "unit": "mg/L"},
    {"name": "FERRITINA",       "ref_low": 30.0,  "ref_high": 400.0, "sex_f_low": 10.0,  "sex_f_high": 200.0, "unit": "ng/mL"},
    {"name": "D_DIMERO",        "ref_low": 0.0,   "ref_high": 500.0, "sex_f_low": 0.0,   "sex_f_high": 500.0, "unit": "ng/mL"},
    {"name": "LACTATO",         "ref_low": 0.5,   "ref_high": 2.2,   "sex_f_low": 0.5,   "sex_f_high": 2.2,   "unit": "mmol/L"},
    {"name": "TGO",             "ref_low": 10.0,  "ref_high": 40.0,  "sex_f_low": 10.0,  "sex_f_high": 35.0,  "unit": "U/L"},
    {"name": "TGP",             "ref_low": 7.0,   "ref_high": 56.0,  "sex_f_low": 7.0,   "sex_f_high": 45.0,  "unit": "U/L"},
    {"name": "BILIRRUBINA_TOT", "ref_low": 0.3,   "ref_high": 1.2,   "sex_f_low": 0.3,   "sex_f_high": 1.2,   "unit": "mg/dL"},
    {"name": "SODIO",           "ref_low": 136.0, "ref_high": 145.0, "sex_f_low": 136.0, "sex_f_high": 145.0, "unit": "mEq/L"},
    {"name": "POTASSIO",        "ref_low": 3.5,   "ref_high": 5.0,   "sex_f_low": 3.5,   "sex_f_high": 5.0,   "unit": "mEq/L"},
    {"name": "GLICOSE",         "ref_low": 70.0,  "ref_high": 99.0,  "sex_f_low": 70.0,  "sex_f_high": 99.0,  "unit": "mg/dL"},
]

PHASES = ["IN", "IN", "IN", "EX", "AB", "AB"]  # Pesos realistas: maioria internado


def _noisy(value: float, noise: float = 0.2) -> float:
    """Adiciona ruído gaussiano a um valor de exame."""
    return round(value * (1 + random.gauss(0, noise)), 2)


def _generate_patient(patient_id: str, sex: str, age: float, n_exams: int, start_date: date) -> dict:
    """Gera um paciente com n_exams medições ao longo de sua internação."""
    exams = []
    current_date = start_date

    # Seleciona aleatoriamente quais exames este paciente teve
    selected = random.sample(EXAM_CATALOG, min(n_exams, len(EXAM_CATALOG)))

    # Simula um paciente com COVID-19: alguns exames fora da referência
    severity = random.random()  # 0=leve, 1=grave

    for exam in selected:
        is_female = sex.upper() == "F"
        ref_low = exam["sex_f_low"] if is_female else exam["ref_low"]
        ref_high = exam["sex_f_high"] if is_female else exam["ref_high"]

        # Pacientes graves tendem a ter valores anormais
        mid = (ref_low + ref_high) / 2
        if severity > 0.6:
            # Valor anormal: fora da faixa de referência
            if random.random() > 0.5:
                value = _noisy(ref_high * random.uniform(1.1, 2.0))
            else:
                value = max(0.0, _noisy(ref_low * random.uniform(0.3, 0.9)))
        else:
            # Valor normal com pequeno ruído
            value = _noisy(mid)

        phase = random.choice(PHASES)
        # Avança a data entre 0 e 2 dias para cada exame
        current_date += timedelta(days=random.randint(0, 2))

        exams.append({
            "exam_name": exam["name"],
            "date": current_date.isoformat(),
            "value": value,
            "phase": phase,
            "ref_low": ref_low,
            "ref_high": ref_high,
            "sex_ref_low": exam["sex_f_low"],
            "sex_ref_high": exam["sex_f_high"],
        })

    return {
        "patient_id": patient_id,
        "sex": sex,
        "age": age,
        "exams": exams,
    }


def generate(n_patients: int = 3, exams_per_patient: int = 8) -> list[dict]:
    """Gera lista de pacientes com exames sintéticos."""
    sexes = ["M", "F", "M", "F", "M", "F", "M"]
    base_date = date(2024, 1, 1)
    patients = []

    for i in range(n_patients):
        pid = f"P{i + 1:03d}"
        sex = sexes[i % len(sexes)]
        age = round(random.uniform(45, 85), 1)
        start = base_date + timedelta(days=i * 30)
        patients.append(_generate_patient(pid, sex, age, exams_per_patient, start))

    return patients


def write_to_dir(patients: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for p in patients:
        path = output_dir / f"{p['patient_id']}.json"
        path.write_text(json.dumps(p, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  criado: {path}")


def send_to_api(patients: list[dict], api_url: str) -> None:
    try:
        import httpx
    except ImportError:
        print("httpx não instalado. Execute: pip install httpx")
        sys.exit(1)

    url = f"{api_url.rstrip('/')}/api/exams/ingest"
    for p in patients:
        try:
            resp = httpx.post(url, json=p, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            print(f"  {p['patient_id']}: risk_score={data.get('risk_score', '?'):.4f}")
        except Exception as e:
            print(f"  ERRO {p['patient_id']}: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera exames sintéticos para o MOSAIC-FL")
    parser.add_argument("--patients", type=int, default=3, help="Número de pacientes (padrão: 3)")
    parser.add_argument("--exams-per-patient", type=int, default=8, help="Exames por paciente (padrão: 8)")
    parser.add_argument("--output-dir", type=Path, default=Path("incoming"), help="Diretório de saída (padrão: ./incoming)")
    parser.add_argument("--send", action="store_true", help="Envia diretamente para a API via HTTP")
    parser.add_argument("--api-url", default="http://localhost:8000", help="URL base da API (padrão: http://localhost:8000)")
    parser.add_argument("--seed", type=int, default=42, help="Semente aleatória (padrão: 42)")
    args = parser.parse_args()

    random.seed(args.seed)

    print(f"Gerando {args.patients} paciente(s) com {args.exams_per_patient} exame(s) cada...")
    patients = generate(args.patients, args.exams_per_patient)

    if args.send:
        print(f"\nEnviando para {args.api_url}...")
        send_to_api(patients, args.api_url)
    else:
        print(f"\nEscrevendo em {args.output_dir}...")
        write_to_dir(patients, args.output_dir)
        print(f"\nPara testar o watcher, copie os arquivos para o volume 'incoming':")
        print(f"  docker compose cp {args.output_dir}/. mosaic-fl-wire-api-1:/app/data/incoming/")
        print(f"\nOu envie diretamente para a API:")
        print(f"  python generate_data.py --send --api-url http://localhost:8000")


if __name__ == "__main__":
    main()
