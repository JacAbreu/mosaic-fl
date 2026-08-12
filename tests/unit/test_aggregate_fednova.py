"""
Testes para aggregate_fednova() — agregação FedNova (Wang et al. 2020) sobre
NDArrays do protocolo Flower, usada pelo Caminho B
(ProductionFedProxStrategy._aggregate_fednova, achado 2026-08-11: até aqui o
Caminho B só agregava por FedAvg/FedProx puro, nunca por FedNova — Seção~
sec:fedprox-fednova-gap do rascunho do TCC).

Espelha o comportamento já validado de aggregate_fednova em
experiments/training/core/fl_core/aggregation.py (Caminho A), reimplementado
sobre NDArrays posicionais em vez de state_dict nomeado.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from mosaicfl.core.federated import aggregate_fednova


class TestAggregateFedNovaSingleClient:
    def test_um_cliente_so_retorna_exatamente_o_estado_dele(self):
        """Com 1 cliente, p_i=1 e tau_eff=tau_i — a fórmula reduz a adotar
        diretamente o estado do único cliente, independente do valor de τ.
        Propriedade usada para justificar (Seção~sec:leave-one-out-caminho-b)
        que o leave-one-out não precisa ser refeito com FedNova: agregação de
        1 cliente é idêntica sob FedAvg e FedNova."""
        w_t = [np.array([1.0, 2.0, 3.0])]
        client_state = [np.array([4.0, 5.0, 6.0])]

        result = aggregate_fednova(w_t, [client_state], num_examples=[100], tau_values=[37])

        np.testing.assert_allclose(result[0], client_state[0])

    def test_um_cliente_varios_valores_de_tau_sempre_igual(self):
        w_t = [np.array([0.0, 0.0])]
        client_state = [np.array([10.0, -10.0])]
        for tau in (1, 5, 100):
            result = aggregate_fednova(w_t, [client_state], num_examples=[50], tau_values=[tau])
            np.testing.assert_allclose(result[0], client_state[0])


class TestAggregateFedNovaDoisClientes:
    def test_mesmo_tau_mesma_amostra_igual_media_simples(self):
        """Com τ e n idênticos entre os 2 clientes, FedNova reduz à média
        simples (mesmo resultado que FedAvg nesse caso degenerado)."""
        w_t = [np.array([0.0, 0.0])]
        client_a = [np.array([2.0, 4.0])]
        client_b = [np.array([6.0, 8.0])]

        result = aggregate_fednova(
            w_t, [client_a, client_b], num_examples=[100, 100], tau_values=[10, 10],
        )
        np.testing.assert_allclose(result[0], [4.0, 6.0])

    def test_tau_igual_entre_clientes_reduz_a_fedavg(self):
        """Propriedade verificável analiticamente: quando τ é o MESMO para
        todos os clientes (só n difere), a fórmula do FedNova se reduz
        exatamente à média ponderada por amostra do FedAvg — τ_eff=T e o termo
        Σp_i·(w_i−w_t)/T, multiplicado de volta por T, cancela o T e sobra
        Σp_i·(w_i−w_t) = Σp_i·w_i − w_t (já que Σp_i=1)."""
        w_t = [np.array([0.0])]
        bpsp = [np.array([10.0])]  # 5,5x mais amostras que o HSL
        hsl = [np.array([1.0])]

        result = aggregate_fednova(
            w_t, [bpsp, hsl], num_examples=[550, 100], tau_values=[7, 7],
        )
        fedavg_equivalent = (550 * 10.0 + 100 * 1.0) / 650
        np.testing.assert_allclose(result[0][0], fedavg_equivalent)

    def test_tau_maior_reduz_a_contribuicao_do_cliente_mesmo_com_n_igual(self):
        """Com n igual entre os dois clientes, só a normalização por τ deveria
        distinguir suas contribuições — o cliente com τ maior (mais passos
        "gastos" pra chegar no mesmo update) pesa proporcionalmente menos por
        unidade de update no termo (w_i−w_t)/τ_i."""
        w_t = [np.array([0.0])]
        client_tau_baixo = [np.array([10.0])]  # τ=1: update "concentrado"
        client_tau_alto = [np.array([10.0])]   # τ=10: mesmo update, mais passos

        result = aggregate_fednova(
            w_t, [client_tau_baixo, client_tau_alto],
            num_examples=[100, 100], tau_values=[1, 10],
        )
        # tau_eff = 0,5*1 + 0,5*10 = 5,5
        # delta = 0,5*10/1 + 0,5*10/10 = 5,0 + 0,5 = 5,5
        # resultado = 0 + 5,5*5,5 = 30,25
        np.testing.assert_allclose(result[0][0], 30.25)


class TestAggregateFedNovaValidacao:
    def test_lista_vazia_de_clientes_levanta_valueerror(self):
        with pytest.raises(ValueError):
            aggregate_fednova([np.array([1.0])], [], num_examples=[], tau_values=[])

    def test_tamanhos_inconsistentes_levanta_valueerror(self):
        w_t = [np.array([1.0])]
        with pytest.raises(ValueError):
            aggregate_fednova(
                w_t, [[np.array([1.0])], [np.array([2.0])]],
                num_examples=[10],  # só 1, mas 2 clientes
                tau_values=[1, 1],
            )

    def test_peso_total_zero_levanta_valueerror(self):
        w_t = [np.array([1.0])]
        with pytest.raises(ValueError):
            aggregate_fednova(w_t, [[np.array([2.0])]], num_examples=[0], tau_values=[1])

    def test_tau_zero_de_um_cliente_nao_quebra_e_nao_zera_a_rodada_toda(self):
        """τ=0 não deveria acontecer na prática (cliente sem nenhum passo), mas
        a função nunca deve levantar ZeroDivisionError — trata como τ=1 (não
        como τ=0) tanto no termo do cliente quanto em τ_eff, senão um único
        cliente degenerado apagaria o progresso de TODOS os clientes da
        rodada (τ_eff colapsaria pra zero se usasse o τ=0 bruto)."""
        w_t = [np.array([0.0])]
        result = aggregate_fednova(w_t, [[np.array([5.0])]], num_examples=[10], tau_values=[0])
        # com 1 cliente só, τ=0 tratado como τ=1 é equivalente a τ=1 de verdade —
        # mesma propriedade de "1 cliente = adota o estado dele" (ver classe acima)
        np.testing.assert_allclose(result[0], [5.0])

    def test_tau_zero_de_um_cliente_entre_varios_nao_derruba_os_outros(self):
        """Com 2 clientes, um deles com τ=0 (degenerado) — o outro cliente,
        saudável, continua contribuindo normalmente; o resultado não colapsa
        pra w_t (o que aconteceria se τ_eff usasse o τ=0 bruto de qualquer
        cliente na soma)."""
        w_t = [np.array([0.0])]
        cliente_ok = [np.array([10.0])]
        cliente_degenerado = [np.array([999.0])]  # update grande, mas τ=0 (deveria pesar como τ=1)

        result = aggregate_fednova(
            w_t, [cliente_ok, cliente_degenerado],
            num_examples=[100, 100], tau_values=[1, 0],
        )
        assert result[0][0] != 0.0  # não colapsou pra w_t


class TestAggregateFedNovaMultiCamada:
    def test_preserva_ordem_e_shape_de_multiplas_camadas(self):
        w_t = [np.zeros((2, 2)), np.zeros(3)]
        client = [np.ones((2, 2)) * 2, np.ones(3) * 3]

        result = aggregate_fednova(w_t, [client], num_examples=[10], tau_values=[5])

        assert result[0].shape == (2, 2)
        assert result[1].shape == (3,)
        np.testing.assert_allclose(result[0], client[0])
        np.testing.assert_allclose(result[1], client[1])

    def test_preserva_dtype_original(self):
        w_t = [np.array([1.0, 2.0], dtype=np.float32)]
        client = [np.array([3.0, 4.0], dtype=np.float32)]
        result = aggregate_fednova(w_t, [client], num_examples=[10], tau_values=[5])
        assert result[0].dtype == np.float32
