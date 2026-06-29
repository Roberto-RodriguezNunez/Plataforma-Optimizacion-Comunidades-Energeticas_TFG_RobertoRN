#!/usr/bin/env bash
# =============================================================================
# entrenar.sh — Entrena + evalúa una familia entera, REANUDABLE tras un corte
# =============================================================================
# Sustituye a run_familia.sh con dos mejoras:
#   1) Salta las versiones YA evaluadas (las que tienen fila en resultados.csv).
#   2) Reanuda las versiones a medias: el runner (multiseed) reutiliza las
#      semillas ya terminadas y reentrena la que quedó cortada (p.ej. apagón).
#
# Tras un corte basta con volver a lanzar el MISMO comando: continúa solo.
#
#   ./experimentos/entrenar.sh dqn            # familia entera (reanudable)
#   ./experimentos/entrenar.sh residual_sac
#   ./experimentos/entrenar.sh G              # solo baselines G1-G4
#
# En WSL/Linux usa .venv/bin/python por defecto (no hace falta PYTHON=...).
# Familias: dqn ppo ppo_continuo td3_residual ddpg_residual residual_sac sac_puro
# =============================================================================
set -euo pipefail

FAM="${1:?uso: entrenar.sh <familia|G>}"
PY="${PYTHON:-.venv/bin/python}"
export PYTHONUNBUFFERED=1
cd "$(dirname "$0")/.."          # raíz del proyecto

CSV="resultados/resultados.csv"

# --- Baselines G1-G4 (no entrena, solo evalúa) ------------------------------
if [[ "$FAM" == "G" ]]; then
  echo "### Baselines G1-G4 ###"
  "$PY" src/eval_suite.py --baselines
  exit 0
fi

# --- Lista de versiones de la familia (desde la registry) -------------------
VERSIONS=$("$PY" -c "from src.training.registro import listar_versiones; print(' '.join(listar_versiones('$FAM')))")
echo "### Familia '$FAM' -> versiones: $VERSIONS ###"

for V in $VERSIONS; do
  # ¿versión ya evaluada? (su fila ya está en el CSV) -> nada que hacer
  if [[ -f "$CSV" ]] && grep -q "^$FAM,$V," "$CSV"; then
    echo "  [HECHA]   $FAM/$V ya está en resultados.csv  ->  salto"
    continue
  fi

  echo "================ ENTRENO  $FAM / $V ================"
  # main_*.py llama a entrenar_multiseed, que REANUDA por dentro:
  #   - semilla terminada  -> la reutiliza (no reentrena)
  #   - semilla a medias    -> la reentrena entera
  #   - semilla no empezada -> la entrena
  "$PY" "src/main_${FAM}.py" --version "$V"

  echo "================ EVAL     $FAM / $V ================"
  "$PY" src/eval_suite.py --familia "$FAM" --version "$V"
done

echo "### Familia '$FAM' COMPLETA  ->  resultados/resultados.csv ###"
