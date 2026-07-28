"""
layer_group.py — Ruído DP diferenciado por grupo de camada (opcional, coexiste com
UniformNoiseStrategy — nunca a substitui).

Motivação (achado 2026-07-27, docs/pesquisa_baseline_implementacao_fontes_
bibliograficas.md, seção 13): BPSP e HSL têm label skew extremo e quase espelhado
(curado_pronto 67,2%×1,5%; melhora_pronto 1,1%×80,6%) — sinal de classe rara já é
estruturalmente frágil antes de qualquer ruído. Hipótese a testar: ruído uniforme
pode estar destruindo esse sinal já fraco especificamente na CABEÇA de classificação,
onde a decisão entre classes acontece — reduzir ruído ali (head_scale<1.0) preserva
mais desse sinal, ao custo de uma garantia de privacidade um pouco mais fraca só
naquela parte do modelo.

Fundamentação bibliográfica (seção 14/7.9 do doc de pesquisa): Li, Zhu & Li,
"Privacy-Preserving Federated Learning with Differential Privacy and Adaptive
Knowledge Distillation for Dynamic Non-IID Data" (ISNCC 2025, DOI
10.1109/ISNCC66965.2025.11250384) propõem DP por camada com pontuação de
sensibilidade S(θ)=α·‖∇θL‖₂+β·I(θ;D)+γ·V(θ), aplicando mecanismos de proteção
diferentes por faixa de sensibilidade (FHE/FE/DP local). Implementar a fórmula
completa exige criptografia homomórfica/funcional (TenSEAL/Pyfhel) — fora de escopo
dado o cronograma do projeto (já registrado como inviável em
project_pressao_cronograma_qualidade). Esta classe implementa a "peça mais barata e
adaptável" já identificada na própria pesquisa bibliográfica: reaproveita só a IDEIA
de modular a magnitude do ruído por camada, com uma escala FIXA e configurável por
grupo (não a pontuação S(θ) contínua e calculada), mantendo o mecanismo gaussiano/RDP
já validado do projeto — sem trocar de mecanismo, sem nova dependência.

Buffers não-treináveis (pos_encoder.pe, positional encoding fixo — ver
src/mosaicfl/core/model.py) são excluídos do ruído inteiramente: não carregam
informação aprendida do dado, não são sensíveis, e receber ruído neles hoje (em
UniformNoiseStrategy) é um efeito colateral do laço `for key in global_state` não
filtrar buffers — mantido lá por fidelidade ao comportamento histórico exato, mas
corrigido aqui porque esta é uma estratégia nova, sem compromisso de paridade byte-a-
byte com o passado.
"""
from .base import DPNoiseStrategy

_EXCLUDED_PREFIXES = ("pos_encoder.",)
_HEAD_PREFIXES = ("classifier.", "pre_classifier.")
_TRANSFORMER_PREFIXES = ("layers.",)
_EMBEDDING_PREFIXES = ("embedding.", "dia_embedding.", "cls_token")


class LayerGroupNoiseStrategy(DPNoiseStrategy):
    def __init__(
        self,
        head_scale: float = 1.0,
        embedding_scale: float = 1.0,
        transformer_scale: float = 1.0,
    ):
        self._head_scale = head_scale
        self._embedding_scale = embedding_scale
        self._transformer_scale = transformer_scale

    def group_for_key(self, key: str) -> str:
        if key.startswith(_EXCLUDED_PREFIXES):
            return "excluded"
        if key.startswith(_HEAD_PREFIXES):
            return "head"
        if key.startswith(_TRANSFORMER_PREFIXES):
            return "transformer"
        if key.startswith(_EMBEDDING_PREFIXES):
            return "embedding"
        return "other"  # chave não prevista — cai no multiplicador base, sem exclusão silenciosa

    def multiplier_for_group(self, group: str, base_noise_multiplier: float) -> float:
        if group == "excluded":
            return 0.0
        if group == "head":
            return base_noise_multiplier * self._head_scale
        if group == "transformer":
            return base_noise_multiplier * self._transformer_scale
        if group == "embedding":
            return base_noise_multiplier * self._embedding_scale
        return base_noise_multiplier
