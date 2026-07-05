# Procedencia de los modelos entrenados — entornos de software

Generado: 2026-07-03. Entorno extraído de `system_info.txt` dentro de cada `.zip` (SB3).

## Entorno de ESTE PC (referencia de reevaluación uniforme)

- Linux-6.18.33.1-microsoft-standard-WSL2-x86_64-with-glibc2.43
- Python 3.14.4 · torch 2.12.1+cu130 · numpy 2.5.0 · SB3 2.9.0

## Agrupación por entorno (clave para la comparabilidad)

### Windows-10 · Python 3.14.6 · PyTorch 2.12.1+cpu  (14 modelos)
- ppo_continuo_PPO-C1, ppo_continuo_PPO-C2, ppo_continuo_PPO-C3, residual_sac_F2-d05, residual_sac_F2-d10, residual_sac_F2-d20, residual_sac_F2-d30, residual_sac_F4-nodawn, residual_sac_F6-s05, residual_sac_F6-s25, residual_sac_SAC-A, residual_sac_SAC-B, residual_sac_SAC-C, sac_puro_F5

### Windows-10 · Python 3.12.7 · PyTorch 2.12.1+cpu  (7 modelos)
- ddpg_residual_DDPG-1 (de _runs), ddpg_residual_DDPG-2 (de _runs), ppo_PPO-D1 (de _runs), ppo_PPO-D2 (de _runs), ppo_PPO-D3 (de _runs), ppo_PPO-D4 (de _runs), ppo_PPO-D5 (de _runs)

### Linux WSL2 · Python 3.14.4 · PyTorch 2.12.1+cu130  (6 modelos)
- dqn_DQN-1, dqn_DQN-2, dqn_DQN-3, dqn_DQN-4, dqn_DQN-5, dqn_DQN-6

### Linux WSL2 · Python 3.12.3 · PyTorch 2.12.1+cpu  (4 modelos)
- td3_residual_TD3-1, td3_residual_TD3-2, td3_residual_TD3-3, td3_residual_TD3-4

## Lectura para la memoria

- **La suite residual completa (SAC-A/B/C, F2-d05/10/20/30, F4-nodawn, F6-s05/s15/s25),
  `sac_puro` y `ppo_continuo` comparten UN mismo entorno** (Windows · Py3.14.6 · torch+cpu).
  → todas las comparaciones INTERNAS de esa suite (barrido δ, ablación DAWN, sensibilidad
  σ del forecast F6, aditivo/mult) son **mismo-entorno y por tanto plenamente comparables**.
- **`td3` (Linux · Py3.12.3 · cpu)** y **`dqn`/`ppo-D`/`ddpg` (este PC · Py3.14 · cu130)**
  están en entornos distintos. La única comparación *ajustada* cruzada es **SAC-C vs TD3-1**;
  las demás (dqn/ppo puros vs MPC) tienen gaps enormes y no se ven afectadas.
- El **eval es determinista** e independiente del entorno (baselines idénticos entre máquinas),
  así que los NÚMEROS son comparables; la varianza de entrenamiento entre entornos se gestiona
  con estadística sobre semillas (IC95 + Wilcoxon), no homogeneizando hardware (Henderson 2017).

## Tabla completa

| modelo | OS | Python | PyTorch (build) |
|---|---|---|---|
| dqn_DQN-1 | Linux WSL2 | 3.14.4 | 2.12.1+cu130 |
| dqn_DQN-2 | Linux WSL2 | 3.14.4 | 2.12.1+cu130 |
| dqn_DQN-3 | Linux WSL2 | 3.14.4 | 2.12.1+cu130 |
| dqn_DQN-4 | Linux WSL2 | 3.14.4 | 2.12.1+cu130 |
| dqn_DQN-5 | Linux WSL2 | 3.14.4 | 2.12.1+cu130 |
| dqn_DQN-6 | Linux WSL2 | 3.14.4 | 2.12.1+cu130 |
| ppo_continuo_PPO-C1 | Windows-10 | 3.14.6 | 2.12.1+cpu |
| ppo_continuo_PPO-C2 | Windows-10 | 3.14.6 | 2.12.1+cpu |
| ppo_continuo_PPO-C3 | Windows-10 | 3.14.6 | 2.12.1+cpu |
| residual_sac_F2-d05 | Windows-10 | 3.14.6 | 2.12.1+cpu |
| residual_sac_F2-d10 | Windows-10 | 3.14.6 | 2.12.1+cpu |
| residual_sac_F2-d20 | Windows-10 | 3.14.6 | 2.12.1+cpu |
| residual_sac_F2-d30 | Windows-10 | 3.14.6 | 2.12.1+cpu |
| residual_sac_F4-nodawn | Windows-10 | 3.14.6 | 2.12.1+cpu |
| residual_sac_F6-s05 | Windows-10 | 3.14.6 | 2.12.1+cpu |
| residual_sac_F6-s25 | Windows-10 | 3.14.6 | 2.12.1+cpu |
| residual_sac_SAC-A | Windows-10 | 3.14.6 | 2.12.1+cpu |
| residual_sac_SAC-B | Windows-10 | 3.14.6 | 2.12.1+cpu |
| residual_sac_SAC-C | Windows-10 | 3.14.6 | 2.12.1+cpu |
| sac_puro_F5 | Windows-10 | 3.14.6 | 2.12.1+cpu |
| td3_residual_TD3-1 | Linux WSL2 | 3.12.3 | 2.12.1+cpu |
| td3_residual_TD3-2 | Linux WSL2 | 3.12.3 | 2.12.1+cpu |
| td3_residual_TD3-3 | Linux WSL2 | 3.12.3 | 2.12.1+cpu |
| td3_residual_TD3-4 | Linux WSL2 | 3.12.3 | 2.12.1+cpu |
| ppo_PPO-D1 (de _runs) | Windows-10 | 3.12.7 | 2.12.1+cpu |
| ppo_PPO-D2 (de _runs) | Windows-10 | 3.12.7 | 2.12.1+cpu |
| ppo_PPO-D3 (de _runs) | Windows-10 | 3.12.7 | 2.12.1+cpu |
| ppo_PPO-D4 (de _runs) | Windows-10 | 3.12.7 | 2.12.1+cpu |
| ppo_PPO-D5 (de _runs) | Windows-10 | 3.12.7 | 2.12.1+cpu |
| ddpg_residual_DDPG-1 (de _runs) | Windows-10 | 3.12.7 | 2.12.1+cpu |
| ddpg_residual_DDPG-2 (de _runs) | Windows-10 | 3.12.7 | 2.12.1+cpu |
