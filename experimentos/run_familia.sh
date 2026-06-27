#!/usr/bin/env bash
# =============================================================================
# run_familia.sh — Entrena + evalúa todas las versiones de una familia
# =============================================================================
# Una sola orden por ordenador. Itera las versiones de la familia (de la
# registry), entrena cada una con las 3 semillas de entreno y luego la evalúa
# con las 10 semillas de ruido, añadiendo la fila a resultados/resultados.csv.
#
# Uso:
#   PYTHON=.venv/bin/python ./experimentos/run_familia.sh dqn
#   ./experimentos/run_familia.sh G                 # solo baselines G1-G4
#   ./experimentos/run_familia.sh residual_sac --solo-eval   # no reentrena
#
# Familias: dqn ppo ppo_continuo td3_residual ddpg_residual residual_sac sac_puro
# (en este entorno 'python' no está en PATH → usa PYTHON=.venv/bin/python)
# =============================================================================
set -euo pipefail

FAM="${1:?uso: run_familia.sh <familia|G> [--solo-eval]}"
MODE="${2:-}"
PY="${PYTHON:-.venv/bin/python}"
export PYTHONUNBUFFERED=1

cd "$(dirname "$0")/.."

if [[ "$FAM" == "G" ]]; then
  echo "### Baselines G1-G4 (10 semillas) ###"
  "$PY" src/eval_suite.py --baselines
  exit 0
fi

VERSIONS=$("$PY" -c "from src.training.registro import listar_versiones; print(' '.join(listar_versiones('$FAM')))")
echo "### Familia '$FAM' -> versiones: $VERSIONS ###"

for V in $VERSIONS; do
  if [[ "$MODE" != "--solo-eval" ]]; then
    echo "================ ENTRENO  $FAM / $V  (3 semillas) ================"
    "$PY" "src/main_${FAM}.py" --version "$V"
  fi
  echo "================ EVAL     $FAM / $V  (10 semillas) ==============="
  "$PY" src/eval_suite.py --familia "$FAM" --version "$V"
done

echo "### Familia '$FAM' COMPLETA -> resultados/resultados.csv ###"
