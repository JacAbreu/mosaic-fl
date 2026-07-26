"""
Cliente Flower com FedProx para treinamento local federado (FedProxClient).

Recebe pesos globais do servidor, treina localmente com DataLoaders de um único hospital,
e devolve apenas os pesos atualizados — nunca os dados brutos.

No modo de simulação, os DataLoaders vêm de prepare_dataloaders_from_db() via
SequencePipeline.build_per_hospital() — um cliente por hospital (HSL, BPSP).
No modo de produção (deploy real), cada instância rodaria em um hospital distinto
com SequencePipeline(hospital_id=FL_CLIENT_ID).build() contra o banco local.

Usa state_dict() para sincronizar todos os tensores (treináveis + buffers de normalização).
A loss local é CrossEntropy + termo proximal FedProx: (μ/2)·‖w_local − w_global‖².
"""
import json
import logging
import time
import numpy as np
import psutil
import torch
import torch.nn.functional as F
import flwr as fl
from torch.utils.data import DataLoader, TensorDataset
from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

from collections import Counter

from mosaicfl.core.model import SimplifiedBEHRT
from .calibration import IsotonicCalibrator, TemperatureScaler
from .config import FED_CFG, MODEL_CFG, RUNTIME_CFG
from .resources import sample_gpu_power_w


def _macro_auc_ovr(probs: torch.Tensor, labels: List[int], num_classes: int) -> Optional[float]:
    """AUC-ROC macro (one-vs-rest), calculado localmente no cliente — mesma filosofia
    de privacidade do F1 (client.py::evaluate): só o escalar agregado sai do hospital,
    nunca probabilidade/rótulo bruto por amostra.

    Diferente de F1 (zero_division=0 é uma convenção válida), AUC não tem convenção
    de "classe ausente" — uma classe com 0 exemplos positivos no val_loader local
    deixa a sub-tarefa binária "classe c vs. resto" matematicamente indefinida, não
    zero. BPSP e HSL têm distribuições de classe muito diferentes (ver
    docs/pesquisa_baseline_implementacao_fontes_bibliograficas.md) — é esperado que
    um cliente não tenha nenhum exemplo de alguma classe rara localmente. Pula essas
    classes em vez de propagar ValueError do sklearn; retorna None (chave omitida na
    agregação, nunca um 0.0 que mascararia "AUC ruim" com "AUC não calculável").
    """
    from sklearn.metrics import roc_auc_score
    y_true = np.asarray(labels)
    y_score = probs.numpy()
    aucs: List[float] = []
    for c in range(num_classes):
        y_true_bin = (y_true == c).astype(int)
        if len(np.unique(y_true_bin)) < 2:
            continue
        try:
            aucs.append(float(roc_auc_score(y_true_bin, y_score[:, c])))
        except ValueError:
            continue
    return float(np.mean(aucs)) if aucs else None


class FedProxClient(fl.client.NumPyClient):
    def __init__(
        self,
        client_id: int,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        loader_factory: Optional[Callable[[str], Tuple[DataLoader, DataLoader]]] = None,
    ):
        """
        train_loader/val_loader: uso direto (simulação/testes) — dados já carregados.
        loader_factory: uso em produção (SGBD) — carregamento adiado para o 1º fit()/
            evaluate(), quando o vocab_json enviado pelo servidor via config já está
            disponível (ver _ensure_data). Recebe o vocab_json (str) e devolve
            (train_loader, val_loader) — quem cacheia entre rounds é o chamador
            (supernode.py), não este cliente (um FedProxClient novo é criado por round).
        """
        self.client_id = client_id
        self.train_loader = train_loader
        self.val_loader = val_loader
        self._loader_factory = loader_factory
        self.vocab: Optional[Dict[str, int]] = None  # setado em _ensure_data(); usado por extract_rag_patterns()
        self.model = SimplifiedBEHRT(use_cls_token=True).to(RUNTIME_CFG.device)
        self._eval_criterion = torch.nn.CrossEntropyLoss()  # sem peso para comparação entre rounds
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=FED_CFG.lr)
        self.global_params: Optional[List[torch.Tensor]] = None
        self.criterion: Optional[torch.nn.Module] = None
        if train_loader is not None:
            self._init_criterion()
        elif loader_factory is None:
            raise ValueError("FedProxClient precisa de train_loader ou loader_factory.")

    def _init_criterion(self) -> None:
        class_weights = self._compute_class_weights(self.train_loader).to(RUNTIME_CFG.device)
        self.criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    def _ensure_data(self, config: Dict) -> None:
        """Carrega train_loader/val_loader sob demanda (produção/SGBD) usando o vocab da rodada.

        Sem isso, o carregamento aconteceria antes de qualquer rodada começar — mas o
        vocabulário canônico só chega via config (vocab_json), disponível apenas quando
        fit()/evaluate() é chamado. Sem vocab_json, falha alto em vez de construir um vocab
        local (que seria incompatível com o de outros clientes — mesmo índice de embedding
        representando tokens diferentes em cada hospital, corrompendo a agregação em silêncio).
        """
        if self.train_loader is not None:
            return
        vocab_json = config.get("vocab_json")
        if not vocab_json:
            raise RuntimeError(
                "Servidor não enviou vocabulário padrão (vocab_json ausente na config da "
                "rodada). Treinamento federado de produção exige vocabulário compartilhado "
                "entre clientes — abortando em vez de construir um vocab local incompatível."
            )
        self.vocab = json.loads(vocab_json)  # usado por extract_rag_patterns() em evaluate()
        self.train_loader, self.val_loader = self._loader_factory(vocab_json)
        self._init_criterion()

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """
        Carrega pesos globais no modelo local via state_dict (treináveis + buffers).
        O zip com state_dict().keys() garante alinhamento posicional com get_parameters().
        """
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        # strict=False permite carregar apenas os parâmetros fornecidos,
        # ignorando buffers que não estão na lista de parâmetros treináveis.
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning("params_missing", extra={"client_id": self.client_id, "keys": missing})
        if unexpected:
            logger.warning("params_unexpected", extra={"client_id": self.client_id, "keys": unexpected})

        # Armazena cópia dos parâmetros globais para o termo proximal
        self.global_params = [p.clone().detach().to(RUNTIME_CFG.device) for p in self.model.parameters()]

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """
        Retorna state_dict completo: parâmetros treináveis + buffers (ex: running_mean do BatchNorm).
        Buffers devem ser sincronizados entre clientes para evitar divergência silenciosa na inferência.
        `.copy()` garante array independente — evita aliasing com a memória dos tensores do modelo.
        """
        return [v.cpu().detach().numpy().copy() for v in self.model.state_dict().values()]

    def _compute_class_weights(self, loader: DataLoader) -> torch.Tensor:
        """
        Pesos inversamente proporcionais à frequência de cada classe no loader local.

        Classes ausentes no conjunto de treino recebem peso 0.0 — não contribuem
        para o gradiente. Usar count=1 como fallback para classes ausentes inflaria
        o peso para valores como total/(n*1) >> peso das classes presentes, o que
        distorce a loss em direção a classes inexistentes.
        """
        counts: Counter = Counter()
        for _, batch_y, *_ in loader:
            counts.update(batch_y.tolist())
        n = MODEL_CFG.num_classes
        total = sum(counts.values()) or 1
        weights = torch.tensor(
            [total / (n * counts[i]) if counts.get(i, 0) > 0 else 0.0 for i in range(n)],
            dtype=torch.float,
        ).clamp(max=15.0)  # teto: peso 47 no BPSP causava explosão de gradiente
        logger.info(
            "class_weights client_id=%s weights=%s counts=%s",
            self.client_id, [round(w, 3) for w in weights.tolist()], dict(counts),
        )
        return weights

    def _proximal_loss(self, loss: torch.Tensor, proximal_mu: float) -> torch.Tensor:
        """Adiciona termo proximal do FedProx com mu recebido do servidor."""
        if self.global_params is None:
            return loss
        proximal_term = 0.0
        for local_w, global_w in zip(self.model.parameters(), self.global_params):
            proximal_term += torch.norm(local_w - global_w, p=2) ** 2
        return loss + (proximal_mu / 2) * proximal_term

    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        self._ensure_data(config)
        local_epochs  = int(config.get("local_epochs",   FED_CFG.local_epochs))
        proximal_mu   = float(config.get("proximal_mu",  FED_CFG.proximal_mu))
        current_round = int(config.get("current_round",  0))

        # Seed por rodada × cliente — garante reprodutibilidade entre runs independentes
        torch.manual_seed(FED_CFG.random_seed + current_round * FED_CFG.num_clients + self.client_id)

        # Amostragem de custo computacional (Caminho B — cada cliente roda em processo/
        # máquina física separada, diferente do Caminho A onde um único processo media
        # tudo). Parametrizável via FL_COLLECT_RESOURCE_METRICS (ligado por padrão neste
        # projeto, mas quem roda o MOSAIC-FL fora deste contexto pode desligar). Quando
        # ligado, cpu_percent(interval=None) precisa de 2 chamadas para ser útil: a 1ª
        # (aqui) calibra e é descartada, a 2ª (fim do fit) retorna a média real do
        # intervalo — ver docs do psutil. Nunca deve interromper o treino por falha de
        # amostragem (GPU ausente é o caso normal em máquinas sem NVIDIA).
        _collect_resources = FED_CFG.collect_resource_metrics
        _proc = psutil.Process() if _collect_resources else None
        if _proc is not None:
            _proc.cpu_percent(interval=None)
        _fit_start = time.time()

        self.set_parameters(parameters)
        self.model.train()
        epoch_losses: List[float] = []
        tau = 0           # passos efetivos reais (batches processados × épocas)
        total_grad_norm = 0.0
        total_batches   = 0

        for epoch in range(local_epochs):
            running_loss = 0.0
            total_samples = 0
            for batch_x, batch_y, batch_dia in self.train_loader:
                try:
                    batch_x   = batch_x.to(RUNTIME_CFG.device)
                    batch_y   = batch_y.to(RUNTIME_CFG.device)
                    batch_dia = batch_dia.to(RUNTIME_CFG.device)
                    self.optimizer.zero_grad()
                    outputs = self.model(batch_x, dia_relativo=batch_dia)
                    loss = self.criterion(outputs, batch_y)
                    loss = self._proximal_loss(loss, proximal_mu)
                    loss.backward()
                    # clipping antes do step — impede explosão de gradiente com pesos de classe altos
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    running_loss    += loss.item() * batch_y.size(0)
                    total_samples   += batch_y.size(0)
                    tau             += 1
                    total_grad_norm += grad_norm.item()
                    total_batches   += 1
                except Exception as e:
                    logger.error("batch_failed", extra={"client_id": self.client_id, "error": str(e)})
                    raise

            epoch_loss = running_loss / total_samples if total_samples > 0 else 0.0
            epoch_losses.append(epoch_loss)

        # DP: clipa update do cliente Δ = w_final − w_inicial à norma S (DP-FedAvg, McMahan et al. 2018)
        dp_update_norm = 0.0
        if FED_CFG.dp_noise_multiplier > 0 and self.global_params is not None:
            with torch.no_grad():
                updates = [p.detach() - g for p, g in zip(self.model.parameters(), self.global_params)]
                update_norm = torch.sqrt(sum(u.norm() ** 2 for u in updates))
                dp_update_norm = update_norm.item()
                scale = min(1.0, FED_CFG.dp_max_grad_norm / (dp_update_norm + 1e-8))
                if scale < 1.0:
                    for param, g in zip(self.model.parameters(), self.global_params):
                        param.data = g + scale * (param.data - g)
            logger.info(
                "dp_update_clip client_id=%s round=%d update_norm=%.4f scale=%.4f",
                self.client_id, current_round, dp_update_norm, min(1.0, FED_CFG.dp_max_grad_norm / (dp_update_norm + 1e-8)),
            )

        avg_loss      = sum(epoch_losses) / len(epoch_losses)
        avg_grad_norm = total_grad_norm / total_batches if total_batches > 0 else 0.0
        logger.info(
            "client_fit client_id=%s loss=%.4f grad_norm=%.4f tau=%d dp_update_norm=%.4f",
            self.client_id, avg_loss, avg_grad_norm, tau, dp_update_norm,
        )

        # flwr.common.Metrics só aceita valores escalares — todos os campos abaixo são
        # float, nunca None (chave omitida quando a amostra não existe, ex: sem GPU,
        # ou quando FL_COLLECT_RESOURCE_METRICS=0 desliga a coleta inteira).
        resource_metrics: Dict = {}
        if _proc is not None:
            _fit_duration_s = time.time() - _fit_start
            _cpu_pct = _proc.cpu_percent(interval=None)
            _ram_mb = _proc.memory_info().rss / (1024 ** 2)
            _gpu_power_w = sample_gpu_power_w()
            resource_metrics = {
                "resource_duration_s": _fit_duration_s,
                "resource_cpu_pct": _cpu_pct,
                "resource_ram_mb": _ram_mb,
            }
            if _gpu_power_w is not None:
                resource_metrics["resource_gpu_power_w"] = _gpu_power_w
                resource_metrics["resource_gpu_energy_wh"] = _gpu_power_w * (_fit_duration_s / 3600)
            logger.info(
                "client_resources client_id=%s duration_s=%.2f cpu_pct=%.1f ram_mb=%.0f gpu_power_w=%s",
                self.client_id, _fit_duration_s, _cpu_pct, _ram_mb,
                f"{_gpu_power_w:.1f}" if _gpu_power_w is not None else "n/a",
            )

        return self.get_parameters(config), total_samples, {
            "loss": avg_loss, "tau": tau, "grad_norm": avg_grad_norm, "dp_update_norm": dp_update_norm,
            **resource_metrics,
        }

    def evaluate(self, parameters: List[np.ndarray], config: Dict) -> Tuple[float, int, Dict]:
        self._ensure_data(config)
        self.set_parameters(parameters)
        self.model.eval()
        correct, total, loss_sum = 0, 0, 0.0
        all_preds: List[int] = []
        all_labels: List[int] = []
        all_logits: List[torch.Tensor] = []
        with torch.no_grad():
            for batch_x, batch_y, batch_dia in self.val_loader:
                batch_x = batch_x.to(RUNTIME_CFG.device)
                batch_y = batch_y.to(RUNTIME_CFG.device)
                batch_dia = batch_dia.to(RUNTIME_CFG.device)
                outputs = self.model(batch_x, dia_relativo=batch_dia)
                loss = self._eval_criterion(outputs, batch_y)
                loss_sum += loss.item() * batch_y.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
                all_preds.extend(predicted.cpu().tolist())
                all_labels.extend(batch_y.cpu().tolist())
                all_logits.append(outputs.detach().cpu())

        accuracy = correct / total if total > 0 else 0
        avg_loss = loss_sum / total if total > 0 else 0.0

        # F1 calculado localmente (nunca expõe predições/labels brutos ao servidor —
        # só os agregados escalares/por-classe, mesma lógica de accuracy/loss).
        # labels=range(num_classes) explícito: garante vetor de mesmo tamanho em
        # todos os clientes, mesmo quando um hospital não tem exemplos de alguma
        # classe local (BPSP e HSL têm distribuições muito diferentes — sem isso,
        # per_class_f1 viria com tamanhos diferentes entre clientes e quebraria
        # a agregação ponderada por classe no servidor).
        from sklearn.metrics import f1_score
        num_classes = MODEL_CFG.num_classes
        if total > 0:
            f1_macro = float(f1_score(
                all_labels, all_preds, average="macro",
                labels=list(range(num_classes)), zero_division=0,
            ))
            per_class_f1 = f1_score(
                all_labels, all_preds, average=None,
                labels=list(range(num_classes)), zero_division=0,
            ).tolist()
        else:
            f1_macro = 0.0
            per_class_f1 = [0.0] * num_classes

        metrics = {
            "accuracy": accuracy,
            "client_id": self.client_id,
            "f1_macro": f1_macro,
            "per_class_f1_json": json.dumps(per_class_f1),
        }

        # AUC-ROC macro (one-vs-rest) — mesmo padrão de privacidade do F1: calculado
        # localmente a partir dos logits já coletados acima (reaproveita o forward
        # pass, não faz um novo), só o escalar agregado sai do hospital. Chave omitida
        # (não 0.0) quando nenhuma classe tem as duas categorias presentes localmente —
        # ver _macro_auc_ovr().
        if total > 0:
            probs = F.softmax(torch.cat(all_logits), dim=-1)
            macro_auc = _macro_auc_ovr(probs, all_labels, num_classes)
            if macro_auc is not None:
                metrics["macro_auc"] = macro_auc

        # Extração de padrões pro RAG — só quando o servidor pede (config, rodada final),
        # nunca em toda rodada: generate_all_profiles() roda o forward com atenção sobre
        # o val_loader inteiro, uma vez por classe — caro pra repetir a cada round.
        # Os "padrões" são perfis por classe de desfecho (top tokens de maior atenção),
        # não registros de paciente — sem patient_id, data ou valor bruto de exame —
        # seguro de enviar ao servidor (mesma filosofia de privacidade de accuracy/F1).
        if config.get("extract_rag_patterns", False) and self.vocab:
            metrics["rag_patterns_json"] = json.dumps(self._extract_rag_patterns())

        # Calibração federada — só quando o servidor pede (config, rodada final, mesmo
        # timing de extract_rag_patterns). Ajusta o calibrador LOCALMENTE (logits já
        # coletados acima, sem forward pass extra) e devolve só estatísticas agregadas/
        # comprimidas (escalar T, ou breakpoints pós-PAV — nunca logits/labels brutos por
        # amostra) — mesma filosofia de privacidade de F1/RAG. Ver
        # docs/pesquisa_baseline_implementacao_fontes_bibliograficas.md §9 (Cormode &
        # Markov / Maddock et al. — calibração ajustada no cliente, agregada no servidor).
        if config.get("calibrate", False) and total > 0:
            metrics.update(self._fit_local_calibrator(
                config.get("calibration_method", "temperature"),
                torch.cat(all_logits),
                torch.tensor(all_labels),
            ))

        return float(avg_loss), total, metrics

    def _fit_local_calibrator(
        self, calibration_method: str, logits: torch.Tensor, labels: torch.Tensor,
    ) -> dict:
        """Ajusta um calibrador local (temperature ou isotonic) sobre o val_loader deste
        cliente e devolve o resultado serializado (compatível com flwr.common.Metrics —
        só escalares/strings, nunca objetos Python brutos)."""
        if calibration_method == "isotonic":
            iso = IsotonicCalibrator().fit_from_logits(logits, labels, num_classes=MODEL_CFG.num_classes)
            logger.info("local_calibration_fit client_id=%s method=isotonic", self.client_id)
            return {
                "calibration_method": "isotonic",
                "isotonic_thresholds_json": json.dumps(iso.export_thresholds()),
            }

        scaler = TemperatureScaler().fit_from_logits(logits, labels, device=str(RUNTIME_CFG.device))
        logger.info("local_calibration_fit client_id=%s method=temperature T=%.4f", self.client_id, scaler.T)
        return {"calibration_method": "temperature", "temperature": scaler.T}

    def _extract_rag_patterns(self) -> list:
        """Gera perfis prototípicos por classe (BEHRTPatternExtractor) sobre o
        val_loader local — nunca inclui dado identificável de paciente."""
        from mosaicfl.core.interpretability import BEHRTPatternExtractor
        extractor = BEHRTPatternExtractor(self.model, self.vocab)
        return extractor.generate_all_profiles(
            self.val_loader, desfechos=list(range(MODEL_CFG.num_classes))
        )


def create_client_fn(
    client_id: int,
    train_data: torch.Tensor,
    train_labels: torch.Tensor,
    val_data: torch.Tensor,
    val_labels: torch.Tensor,
) -> FedProxClient:
    """Stub de compatibilidade — delega para experiments.training.core.dataloaders.create_synthetic_client.

    Exclusivo para simulações com dados sintéticos e testes unitários.
    No pipeline com dados reais (data_source != 'synthetic'), os loaders vêm de
    prepare_dataloaders_from_db() e são passados diretamente ao FedProxClient — esta
    função não participa desse fluxo.
    """
    from experiments.training.core.dataloaders import create_synthetic_client
    return create_synthetic_client(client_id, train_data, train_labels, val_data, val_labels)