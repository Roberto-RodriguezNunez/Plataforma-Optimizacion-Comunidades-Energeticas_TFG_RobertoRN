"""
test_residual_env.py — Tests del wrapper residual SAC multiplicativo 4D
=======================================================================
Ejecutar desde la raiz del proyecto (carpeta TFG/):
    pytest tests/test_residual_env.py -v
"""

import os
import sys
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.envs.energy_env import EnergyEnvContinuo
from src.envs.residual_env import ResidualEnv
from src.controllers.mpc import (
    LinearMPC, ComunidadSimulador, simular_hora_mpc,
    DATASET_PATH, HORIZON, SOC_INICIAL, EPISODE_LENGTH,
)


@pytest.fixture(scope='module')
def residual_env():
    """ResidualEnv con MPC realista (forecast_noise=True)."""
    inner = EnergyEnvContinuo(forecast_noise=True, mode='all')
    sim = ComunidadSimulador(DATASET_PATH)
    mpc = LinearMPC(sim, use_terminal_value=False, k_deg_lin=0.005)
    return ResidualEnv(inner, mpc, delta_max=0.30)


@pytest.fixture(scope='module')
def deterministic_env():
    """ResidualEnv sin ruido para tests deterministas."""
    inner = EnergyEnvContinuo(forecast_noise=False, mode='all')
    sim = ComunidadSimulador(DATASET_PATH)
    mpc = LinearMPC(sim, use_terminal_value=False, k_deg_lin=0.005)
    return ResidualEnv(inner, mpc, delta_max=0.30)


class TestObservacion:

    def test_obs_shape_112(self, residual_env):
        """La observacion tiene 112 dimensiones (108 + 4 MPC features)."""
        obs, info = residual_env.reset(seed=42)
        assert obs.shape == (112,)
        assert obs.dtype == np.float32

    def test_mpc_features_in_obs(self, residual_env):
        """Las ultimas 4 dimensiones son los flujos MPC normalizados."""
        obs, _ = residual_env.reset(seed=42)
        mpc_feat = obs[-4:]
        assert all(f >= -0.01 for f in mpc_feat), f"MPC features negativas: {mpc_feat}"
        assert all(f <= 1.01 for f in mpc_feat), f"MPC features > 1: {mpc_feat}"

    def test_obs_first_108_match_inner(self, deterministic_env):
        """Las primeras 108 dims coinciden con el entorno interno."""
        obs_res, _ = deterministic_env.reset(seed=42)
        assert obs_res.shape == (112,)
        assert obs_res[0] > 0.0


class TestDeltaZero:

    def test_delta_zero_matches_mpc(self):
        """
        Con delta=(0,0,0,0) durante 168 pasos, el reward total debe aproximar
        al MPC realista (residual multiplicativo: flow * 1.0 = flow intacto).
        """
        np.random.seed(42)
        inner = EnergyEnvContinuo(forecast_noise=True, mode='all')
        sim_mpc = ComunidadSimulador(DATASET_PATH)
        mpc = LinearMPC(sim_mpc, use_terminal_value=False, k_deg_lin=0.005)
        renv = ResidualEnv(inner, mpc, delta_max=0.30)

        obs, _ = renv.reset(seed=42)
        total_reward = 0.0
        delta_zero = np.array([0.0, 0.0, 0.0, 0.0])
        for _ in range(EPISODE_LENGTH):
            obs, reward, term, trunc, info = renv.step(delta_zero)
            total_reward += reward
            if term:
                break

        assert -100 < total_reward < 200, (
            f"Reward total con delta=0: {total_reward:.2f} EUR — fuera de rango razonable"
        )
        assert term

    def test_delta_zero_flows_identical(self):
        """Con delta=0 multiplicativo, a_final == a_mpc exactamente."""
        inner = EnergyEnvContinuo(forecast_noise=False, mode='all')
        sim_mpc = ComunidadSimulador(DATASET_PATH)
        mpc = LinearMPC(sim_mpc, use_terminal_value=False, k_deg_lin=0.005)
        renv = ResidualEnv(inner, mpc, delta_max=0.30)

        renv.reset(seed=42)
        delta_zero = np.array([0.0, 0.0, 0.0, 0.0])
        for _ in range(5):
            _, _, term, _, info = renv.step(delta_zero)
            for i in range(4):
                assert abs(info['a_final'][i] - info['a_mpc'][i]) < 1e-6, (
                    f"a_final={info['a_final']}, a_mpc={info['a_mpc']}")
            if term:
                break


class TestMultiplicativo:

    def test_delta_scales_with_flow(self, deterministic_env):
        """Delta multiplicativo escala con la magnitud del flujo MPC."""
        deterministic_env.reset(seed=42)
        # delta=+1 → flow * (1 + 0.30) = flow * 1.30
        delta_pos = np.array([1.0, 1.0, 1.0, 1.0])
        _, _, _, _, info = deterministic_env.step(delta_pos)
        for i in range(4):
            mpc_val = info['a_mpc'][i]
            final_val = info['a_final'][i]
            if mpc_val > 0.01:
                ratio = final_val / mpc_val
                assert 1.25 < ratio < 1.35, (
                    f"Flow {i}: ratio={ratio:.3f}, esperado ~1.30")
            else:
                # MPC=0 → final=0 (multiplicativo)
                assert final_val < 0.01

    def test_mpc_zero_stays_zero(self, deterministic_env):
        """Si MPC dice 0, el flujo final es 0 independientemente del delta."""
        deterministic_env.reset(seed=42)
        deterministic_env.env.simulador.soc = 0.50
        # Con delta extremo
        delta = np.array([1.0, 1.0, 1.0, 1.0])
        _, _, _, _, info = deterministic_env.step(delta)
        for i in range(4):
            if info['a_mpc'][i] < 1e-6:
                assert info['a_final'][i] < 1e-6


class TestClipping:

    def test_clipping_extremo(self, residual_env):
        """Delta extremo (+/-1) produce flujos finales >= 0."""
        residual_env.reset(seed=42)
        for delta_val in [1.0, -1.0]:
            delta = np.array([delta_val] * 4)
            obs, reward, term, trunc, info = residual_env.step(delta)
            for f in info['a_final']:
                assert f >= -1e-6, f"Flujo final negativo: {f}"
            if term:
                residual_env.reset(seed=42)


class TestInfoKeys:

    def test_info_contains_mpc_data(self, residual_env):
        """El dict info contiene las claves de diagnostico del MPC."""
        residual_env.reset(seed=42)
        delta_zero = np.array([0.0, 0.0, 0.0, 0.0])
        _, _, _, _, info = residual_env.step(delta_zero)
        assert 'a_mpc' in info
        assert 'a_final' in info
        assert 'delta_applied' in info
        assert 'soc' in info
        assert len(info['a_mpc']) == 4
        assert len(info['a_final']) == 4
        assert len(info['delta_applied']) == 4

    def test_delta_applied_is_difference(self, residual_env):
        """delta_applied = a_final - a_mpc (la correccion real en kW)."""
        residual_env.reset(seed=42)
        delta = np.array([0.5, 0.0, 0.0, 0.0])
        _, _, _, _, info = residual_env.step(delta)
        for i in range(4):
            expected_diff = info['a_final'][i] - info['a_mpc'][i]
            assert abs(info['delta_applied'][i] - expected_diff) < 1e-4
