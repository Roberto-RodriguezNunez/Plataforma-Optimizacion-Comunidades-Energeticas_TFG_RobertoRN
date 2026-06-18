"""
edge_loop.py — Bucle de operación en hardware de borde (edge)
=============================================================
Por cada hora simulada:
  1. Solicita ventana de datos al DataFeed (emulado o real).
  2. Resuelve OnnxResidualController (sin SB3 ni torch).
  3. Avanza el estado físico del simulador.
  4. Emite la acción y métricas a stdout como JSON.
  5. Si SAAS_API_URL/EDGE_API_KEY/COMUNIDAD_ID definidos, POST al SaaS.

Configuración por variables de entorno:
  FEED_MODE    : 'historico' | 'simulado'  (default: historico)
  ONNX_PATH    : ruta al .onnx              (default: models/residual_sac_actor.onnx)
  NPZ_PATH     : ruta al .npz              (default: models/vec_normalize_v5_1M.npz)
  DELTA_MAX    : float                      (default: 0.15)
  N_STEPS      : enteros ≥ 1               (default: 2160 — tres meses)
  START_STEP   : índice de inicio en el dataset (default: 0)
  START_DATE   : fecha de inicio simulada ISO-8601 (default: 2025-01-01)
                 Cada step avanza 1 hora → step 0 = START_DATE 00:00 UTC
  SAAS_API_URL : URL base del SaaS Flask (ej. http://web:5000) — opcional
  EDGE_API_KEY : clave API para autenticar POST al SaaS — opcional
  COMUNIDAD_ID : id numérico de la comunidad en el SaaS — opcional

Uso local:
    python -m src.production.edge_loop
    FEED_MODE=simulado python -m src.production.edge_loop
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import yaml

try:
    import requests as _requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.api.data_feed import get_feed
from src.benchmarks.mpc_benchmark import (
    ComunidadSimulador,
    DATASET_PATH,
    LinearMPC,
    simular_hora_mpc,
    SOC_INICIAL,
    EPISODE_LENGTH,
    HORIZON,
)
from src.production.onnx_inference import OnnxResidualController

# ── Config ────────────────────────────────────────────────────────────────────
_CFG_PATH = os.path.join(ROOT, 'config', 'system.yaml')
with open(_CFG_PATH, 'r', encoding='utf-8') as _f:
    _CFG = yaml.safe_load(_f)

_PROD_CFG = _CFG.get('produccion', {})
_MPC_CFG  = _CFG['mpc']
_TV_CFG   = _MPC_CFG['valor_terminal']

_DEFAULT_ONNX      = os.path.join(ROOT, _PROD_CFG.get('onnx_path', 'models/residual_sac_actor.onnx'))
_DEFAULT_NPZ       = os.path.join(ROOT, _PROD_CFG.get('npz_path',  'models/vec_normalize_v5_1M.npz'))
_DEFAULT_DELTA_MAX = float(_PROD_CFG.get('delta_max', 0.15))


# ── Edge loop ─────────────────────────────────────────────────────────────────

def run(
    n_steps: int | None = None,
    start_step: int = 0,
    feed_mode: str | None = None,
    onnx_path: str | None = None,
    npz_path: str | None = None,
    delta_max: float | None = None,
) -> None:
    """Ejecuta el bucle de operación edge durante n_steps horas."""
    n_steps    = n_steps   or int(os.environ.get('N_STEPS', 2160))  # 3 meses por defecto
    start_step = start_step or int(os.environ.get('START_STEP', 0))
    onnx_path  = onnx_path  or os.environ.get('ONNX_PATH', _DEFAULT_ONNX)
    npz_path   = npz_path   or os.environ.get('NPZ_PATH',  _DEFAULT_NPZ)
    delta_max  = delta_max  or float(os.environ.get('DELTA_MAX', _DEFAULT_DELTA_MAX))

    # Timestamp simulado: START_DATE + 1 hora por step
    _start_date_str = os.environ.get('START_DATE', '2025-01-01')
    _sim_t0 = datetime.fromisoformat(_start_date_str).replace(tzinfo=timezone.utc)

    # DataFeed
    feed = get_feed(feed_mode)
    sim  = feed.sim

    # MPC + controlador ONNX
    mpc = LinearMPC(
        sim,
        use_terminal_value=_TV_CFG['activado'],
        terminal_lambda=_TV_CFG['lambda'],
        terminal_price_mode=_TV_CFG['modo_precio'],
        k_deg_lin=_MPC_CFG['k_deg_lin'],
    )
    ctrl = OnnxResidualController(
        mpc=mpc,
        sim=sim,
        onnx_path=onnx_path,
        npz_path=npz_path,
        delta_max=delta_max,
    )

    # Config de integración SaaS (opcional)
    saas_url     = os.environ.get('SAAS_API_URL', '').rstrip('/')
    edge_key     = os.environ.get('EDGE_API_KEY', '')
    comunidad_id = os.environ.get('COMUNIDAD_ID', '')
    _post_saas   = bool(saas_url and edge_key and comunidad_id and _HAS_REQUESTS)

    _auth_headers = {'Authorization': f'Bearer {edge_key}'}

    def _publicar_cierre(mes: str) -> None:
        """Llama al SaaS para generar el CierreMensual del mes indicado."""
        try:
            r = _requests.post(
                f'{saas_url}/api/edge/generar-cierre',
                json={'comunidad_id': int(comunidad_id), 'mes': mes},
                headers=_auth_headers,
                timeout=60,
            )
            data = r.json() if r.content else {}
            print(json.dumps({'event': 'cierre_publicado', 'mes': mes,
                              'n_viviendas': data.get('n_viviendas', 0),
                              'msg': data.get('msg', '')}), flush=True)
        except Exception as exc:
            print(f'[edge] WARN cierre {mes} falló: {exc}', file=sys.stderr)

    # Esperar a que el SaaS esté disponible antes de empezar
    if _post_saas:
        import time as _time
        print(json.dumps({'event': 'waiting_saas', 'url': saas_url}), flush=True)
        for _t in range(24):  # hasta 2 minutos (24 × 5 s)
            try:
                r = _requests.get(f'{saas_url}/health', timeout=3)
                if r.status_code == 200:
                    print(json.dumps({'event': 'saas_ready'}), flush=True)
                    break
            except Exception:
                pass
            _time.sleep(5)
        else:
            print('[edge] WARN SaaS no disponible tras 120 s, continuando sin integración',
                  file=sys.stderr)
            _post_saas = False

    # Estado inicial
    sim.current_step = start_step
    sim.soc = SOC_INICIAL
    ben_total  = 0.0
    mes_actual = _sim_t0.strftime('%Y-%m')

    print(json.dumps({'event': 'start', 'n_steps': n_steps,
                      'feed_mode': os.environ.get('FEED_MODE', 'historico'),
                      'delta_max': delta_max, 'start_step': start_step,
                      'start_date': _start_date_str,
                      'saas_integration': _post_saas}),
          flush=True)

    for i in range(n_steps):
        step     = sim.current_step
        sim_ts   = _sim_t0 + timedelta(hours=i)
        sim_ts_s = sim_ts.isoformat()
        mes_step = sim_ts.strftime('%Y-%m')

        # Detectar cambio de mes → publicar cierre del mes que acaba de cerrar
        if _post_saas and mes_step != mes_actual:
            _publicar_cierre(mes_actual)
            mes_actual = mes_step

        forecast = feed.get_window(step, HORIZON)
        state    = {'soc': sim.soc, 'step': step}

        row           = forecast[0]
        consumo_total = float(row[0])
        gen_total     = float(row[1])
        precio_compra = float(row[2])
        precio_exc    = float(row[3])

        action = ctrl.solve(state, forecast)

        b, bm = simular_hora_mpc(
            sim,
            action['P_carga_solar'],
            action['P_carga_red'],
            action['P_descarga_casa'],
            action['P_descarga_red'],
        )
        ben_total += bm

        record = {
            'step':               step,
            'ts':                 sim_ts_s,
            'soc':                round(sim.soc, 4),
            'consumo_total_kwh':  round(consumo_total, 4),
            'gen_total_kwh':      round(gen_total, 4),
            'precio_compra':      round(precio_compra, 5),
            'precio_exc':         round(precio_exc, 5),
            'P_carga_solar':      round(action['P_carga_solar'],   3),
            'P_carga_red':        round(action['P_carga_red'],     3),
            'P_descarga_casa':    round(action['P_descarga_casa'], 3),
            'P_descarga_red':     round(action['P_descarga_red'],  3),
            'beneficio_marginal': round(bm, 4),
            'beneficio_acum':     round(ben_total, 4),
        }
        print(json.dumps(record), flush=True)

        if _post_saas:
            payload = dict(record)
            payload['comunidad_id'] = int(comunidad_id)
            del payload['beneficio_acum']
            try:
                _requests.post(
                    f'{saas_url}/api/edge/decision',
                    json=payload,
                    headers=_auth_headers,
                    timeout=5,
                )
            except Exception as exc:
                print(f'[edge] WARN POST SaaS falló step={step}: {exc}', file=sys.stderr)

    # Publicar cierre del último mes al terminar
    if _post_saas:
        _publicar_cierre(mes_actual)

    print(json.dumps({
        'event': 'done',
        'n_steps': n_steps,
        'beneficio_marginal_total': round(ben_total, 4),
    }), flush=True)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    run()
