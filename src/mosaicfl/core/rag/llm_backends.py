"""
llm_backends.py — Backends de geração de texto para a justificativa clínica do RAG.

huggingface — AutoModelForCausalLM (padrão: distilgpt2), roda no processo Python.
ollama      — POST localhost:11434/api/generate (ex: gemma3:4b), processo separado.
"""
import json
import urllib.error
import urllib.request

import torch


def _check_ollama_available(timeout: int = 5) -> bool:
    """Testa se o Ollama está acessível em localhost:11434. Retorna False se não estiver."""
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=timeout):
            return True
    except Exception:
        return False


def _generate_ollama(model: str, prompt: str, max_tokens: int, timeout: int = 120) -> str:
    """Gera texto via Ollama (POST localhost:11434/api/generate). Sem dependências externas."""
    payload = json.dumps({
        "model":  model,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": max_tokens, "temperature": 0.7},
    }).encode()
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())["response"].strip()
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "Ollama não está acessível em localhost:11434. "
            "Inicie o Ollama com 'ollama serve' e verifique se o modelo está disponível "
            f"com 'ollama pull {model}'."
        ) from exc


def _load_huggingface_backend(llm_model: str):
    """Carrega tokenizer + pipeline HuggingFace. Retorna (tokenizer, generator)."""
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline as hf_pipeline
    tokenizer = AutoTokenizer.from_pretrained(llm_model)
    tokenizer.clean_up_tokenization_spaces = False
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    _llm = AutoModelForCausalLM.from_pretrained(llm_model)
    generator = hf_pipeline(
        "text-generation",
        model=_llm,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1,
        truncation=True,
    )
    return tokenizer, generator
