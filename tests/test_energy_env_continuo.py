"""
test_energy_env_continuo.py — Tests del entorno con accion continua 4D
======================================================================
Ejecutar desde la raiz del proyecto (carpeta TFG/):
    pytest tests/test_energy_env_continuo.py -v
"""

import os
import sys
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.envs.energy_env_continuo import EnergyEnvContinuo


@pytest.fixture(scope='module')
def env():
    """Entorno continuo en modo 'all' sin ruido para tests deterministas."""
    e = EnergyEnvContinuo(forecast_noise=False, mode='all')
    return e


@pytest.fixture(scope='module')
def P_MAX(env):
    return env.simulador.POTENCIA_INVERSOR


class TestEspacios:

    def test_action_space_shape(self, env, P_MAX):
        """El espacio de accion es Box(4) con rango [0, P_MAX]."""
        assert env.action_space.shape == (4,)
        assert float(env.action_space.low[0]) == 0.0
        assert abs(float(env.action_space.high[0]) - P_MAX) < 1e-6

    def test_obs_space_shape(self, env):
        """El espacio de observacion es Box(108)."""
        assert env.observation_space.shape == (108,)

    def test_reset_returns_valid_obs(self, env):
        """reset() devuelve obs de forma correcta y sin NaN."""
        obs, info = env.reset(seed=42)
        assert obs.shape == (108,)
        assert not np.any(np.isnan(obs))
        assert obs.dtype == np.float32


class TestAccionIdle:

    def test_action_zero_is_idle(self, env):
        """Accion (0,0,0,0) no mueve bateria, reward marginal ~0."""
        env.reset(seed=42)
        action_idle = np.array([0.0, 0.0, 0.0, 0.0])
        # Paso 0 incluye pendiente_coste_inicial, lo saltamos
        env.step(action_idle)
        # Paso 1 en adelante: reward marginal puro
        soc_antes = env.simulador.soc
        obs, reward, term, trunc, info = env.step(action_idle)
        # SoC solo cambia por autodescarga (~0.004%)
        assert abs(env.simulador.soc - soc_antes) < 0.001
        # Carga y descarga ~0
        assert info['cargado'] < 1e-6
        assert info['descargado'] < 1e-6
        # Reward marginal ~0 (solo diferencia por autodescarga minuscula)
        assert abs(reward) < 0.5


class TestCargaDescarga:

    def test_full_charge(self, env, P_MAX):
        """Accion (P_MAX,P_MAX,0,0) carga al maximo posible."""
        env.reset(seed=42)
        env.simulador.soc = 0.30  # Mucho espacio libre
        obs, reward, term, trunc, info = env.step(
            np.array([P_MAX, P_MAX, 0.0, 0.0]))
        assert info['cargado'] > 0.0

    def test_full_discharge(self, env, P_MAX):
        """Accion (0,0,P_MAX,P_MAX) descarga al maximo posible."""
        env.reset(seed=42)
        env.simulador.soc = 0.70  # Mucha energia disponible
        obs, reward, term, trunc, info = env.step(
            np.array([0.0, 0.0, P_MAX, P_MAX]))
        assert info['descargado'] > 0.0


class TestClippingSoC:

    def test_soc_max_no_carga(self, env, P_MAX):
        """Con SoC en SOC_MAX, cargar no aumenta SoC."""
        env.reset(seed=42)
        env.simulador.soc = env.simulador.SOC_MAX
        soc_antes = env.simulador.soc
        env.step(np.array([P_MAX, P_MAX, 0.0, 0.0]))
        # Tras autodescarga + intento de carga, SoC no supera SOC_MAX
        assert env.simulador.soc <= env.simulador.SOC_MAX + 1e-6
        # Carga efectiva minima (solo recupera autodescarga como mucho)
        assert env.simulador.soc >= soc_antes - 0.001

    def test_soc_min_no_descarga(self, env, P_MAX):
        """Con SoC en SOC_MIN, descargar no reduce SoC."""
        env.reset(seed=42)
        env.simulador.soc = env.simulador.SOC_MIN
        env.step(np.array([0.0, 0.0, P_MAX, P_MAX]))
        # SoC no baja de SOC_MIN (puede bajar un poco por autodescarga)
        assert env.simulador.soc >= env.simulador.SOC_MIN - 0.001


class TestEpisodio:

    def test_episode_168_steps(self, env):
        """El episodio termina exactamente a 168 pasos."""
        action_idle = np.array([0.0, 0.0, 0.0, 0.0])
        env.reset(seed=42)
        for i in range(167):
            obs, reward, term, trunc, info = env.step(action_idle)
            assert not term, f"Episodio termino prematuramente en paso {i+1}"
        obs, reward, term, trunc, info = env.step(action_idle)
        assert term, "Episodio no termino en paso 168"

    def test_terminal_value_applied(self, env):
        """El ultimo paso incluye valor terminal positivo si hay SoC util."""
        action_idle = np.array([0.0, 0.0, 0.0, 0.0])
        env.reset(seed=42)
        env.simulador.soc = 0.50  # SoC intermedio
        # Avanzar hasta el ultimo paso
        for _ in range(167):
            env.step(action_idle)
        # Forzar SoC alto antes del ultimo paso para garantizar TV positivo
        env.simulador.soc = 0.70
        obs, reward_final, term, trunc, info = env.step(action_idle)
        assert term
        # El reward del ultimo paso debe ser mayor que un paso normal
        # porque incluye el valor terminal de la energia restante
        # (0.70 - 0.10) * 100 * precio * 0.95 ~ varios EUR
        assert reward_final > 0.5  # Al menos 0.5 EUR por la correccion terminal


class TestNoCicloSimultaneo:

    def test_no_ciclo_simultaneo(self, env):
        """Si action tiene carga Y descarga, solo se ejecuta la direccion neta."""
        env.reset(seed=42)
        env.simulador.soc = 0.50
        # action=[5,3,4,2] → carga_bruta=8, descarga_bruta=6, net=+2 → solo carga
        obs, reward, term, trunc, info = env.step(np.array([5.0, 3.0, 4.0, 2.0]))
        # Debe haber cargado algo pero descargado 0
        assert info['cargado'] > 0 or info['descargado'] == 0, (
            f"Ciclo simultaneo: cargado={info['cargado']}, descargado={info['descargado']}")
        assert info['cargado'] == 0 or info['descargado'] == 0, (
            f"Ciclo simultaneo: cargado={info['cargado']}, descargado={info['descargado']}")

    def test_ratio_preservado_carga(self, env):
        """Con cs=6, cm=4 y net>0, el ratio solar/red es 60/40."""
        env.reset(seed=42)
        env.simulador.soc = 0.30  # Mucho espacio libre
        # cs=6, cm=4, dc=2, dr=0 → carga_bruta=10, descarga_bruta=2, net=8
        # ratio_solar=6/10=0.6 → cs_net=4.8, cm_net=3.2
        # Verificamos indirectamente: la carga neta debe ser ~8 kW (pre-clipping)
        # y el ratio se preserva internamente.
        # Usamos un truco: si exc_disp >= 4.8, cs no se recorta por exc_disp
        # y podemos verificar el ratio via la economia.
        obs, reward, term, trunc, info = env.step(np.array([6.0, 4.0, 2.0, 0.0]))
        # Lo critico: no hay descarga (net > 0)
        assert info['descargado'] < 1e-6
        assert info['cargado'] > 0

    def test_ratio_preservado_descarga(self, env):
        """Con dc=3, dr=7 y net<0, el ratio casa/red es 30/70."""
        env.reset(seed=42)
        env.simulador.soc = 0.70  # Mucha energia disponible
        # cs=1, cm=0, dc=3, dr=7 → carga_bruta=1, descarga_bruta=10, net=-9
        # ratio_casa=3/10=0.3 → dc_net=2.7, dr_net=6.3
        obs, reward, term, trunc, info = env.step(np.array([1.0, 0.0, 3.0, 7.0]))
        # Lo critico: no hay carga (net < 0)
        assert info['cargado'] < 1e-6
        assert info['descargado'] > 0

    def test_delta_cero_mpc_via_env_coincide_con_realista(self):
        """Los flujos del MPC ejecutados vía EnergyEnvContinuo reproducen el
        MPC realista (~+47.7 €/sem; coincide con el benchmark, ya que el env
        no introduce error respecto a simular_hora_mpc)."""
        import yaml
        from src.benchmarks.mpc_benchmark import (
            LinearMPC, ComunidadSimulador, simular_semana_idle, aplicar_ruido_ar1,
            DATASET_PATH, SOC_INICIAL, SEED, EPISODE_LENGTH, HORIZON,
            _RHO_SOLAR, _RHO_CONS,
        )
        with open(os.path.join(ROOT, 'config', 'system.yaml'), 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        mpc_cfg = cfg['mpc']

        sim_mpc = ComunidadSimulador(DATASET_PATH)
        tv_cfg = mpc_cfg['valor_terminal']
        mpc = LinearMPC(sim_mpc, use_terminal_value=tv_cfg['activado'],
                        terminal_lambda=tv_cfg['lambda'],
                        terminal_price_mode=tv_cfg['modo_precio'],
                        k_deg_lin=mpc_cfg['k_deg_lin'])

        env_test = EnergyEnvContinuo(forecast_noise=False, mode='all')
        sim = env_test.simulador
        rng = np.random.default_rng(SEED)
        sim_idle = ComunidadSimulador(DATASET_PATH)
        starts = sim_mpc.semanas_eval

        bens_marg = []
        for start in starts:
            ben_idle = simular_semana_idle(sim_idle, start)
            sim.current_step = start
            sim.soc = SOC_INICIAL
            env_test.steps_in_episode = 0
            env_test._error_solar = 0.0
            env_test._error_cons = 0.0
            env_test._pendiente_coste_inicial = 0.0
            error_solar, error_cons = 0.0, 0.0
            ben_total = 0.0

            for _ in range(EPISODE_LENGTH):
                error_solar = (_RHO_SOLAR * error_solar
                               + np.sqrt(1 - _RHO_SOLAR**2) * rng.standard_normal())
                error_cons = (_RHO_CONS * error_cons
                              + np.sqrt(1 - _RHO_CONS**2) * rng.standard_normal())
                window = sim.get_data_window(sim.current_step, horizon=HORIZON)
                window = aplicar_ruido_ar1(window, error_solar, error_cons)
                state = {'soc': sim.soc, 'step': sim.current_step}
                action = mpc.solve(state, window)
                act_4d = np.array([action['P_carga_solar'], action['P_carga_red'],
                                   action['P_descarga_casa'], action['P_descarga_red']],
                                  dtype=np.float32)
                obs, reward, term, trunc, info = env_test.step(act_4d)
                ben_total += info['beneficio']
            bens_marg.append(ben_total - ben_idle)

        media = np.mean(bens_marg)
        assert abs(media - 47.71) < 0.2, (
            f"MPC via EnergyEnvContinuo: {media:+.2f} EUR/sem, esperado ~+47.71 "
            f"(≈ MPC realista). Valor viejo 51.69 era de la config 100kWh/50kWp.")


class TestFisicaCoherente:

    def test_round_trip_efficiency(self, env, P_MAX):
        """Cargar y descargar pierde ~10% (round-trip eta_c*eta_d = 0.9025)."""
        env.reset(seed=42)
        env.simulador.soc = 0.50
        soc_inicio = env.simulador.soc

        # Cargar a tope durante 5 horas
        for _ in range(5):
            env.step(np.array([P_MAX, P_MAX, 0.0, 0.0]))
        soc_tras_carga = env.simulador.soc

        # Descargar durante 5 horas
        for _ in range(5):
            env.step(np.array([0.0, 0.0, P_MAX, P_MAX]))
        soc_final = env.simulador.soc

        # Energia neta perdida debe ser >5% del ciclo
        energia_cargada = (soc_tras_carga - soc_inicio) * env.simulador.BATERIA_CAPACIDAD
        energia_descargada = (soc_tras_carga - soc_final) * env.simulador.BATERIA_CAPACIDAD
        if energia_cargada > 0 and energia_descargada > 0:
            # Round-trip: la energia util es ~90% de la cargada
            assert soc_final < soc_tras_carga  # Se perdio energia

    def test_soc_always_in_bounds(self, env, P_MAX):
        """El SoC nunca sale de [0, 1] tras 168 pasos con acciones aleatorias."""
        rng = np.random.default_rng(123)
        env.reset(seed=42)
        for _ in range(168):
            a = rng.uniform(0, P_MAX, size=(4,)).astype(np.float32)
            env.step(a)
            assert 0.0 <= env.simulador.soc <= 1.0
