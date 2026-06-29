"""
export_onnx.py — Exportación del actor SAC a ONNX y volcado de VecNormalize
=============================================================================
Genera dos artefactos en models/:
  - residual_sac_actor.onnx   : actor determinista (tanh(mean)), opset 17
  - vec_normalize_v5_1M.npz   : estadísticas de normalización sin SB3

Requiere: torch, stable-baselines3, onnx
NO incluir en el contenedor edge (solo se ejecuta una vez en el entorno de
entrenamiento).

Uso:
    python -m src.production.export_onnx
"""

import os
import sys
import pickle

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.core.obs_builder import OBS_DIM_RESIDUAL

# ── Rutas (defaults sobreescribibles por CLI) ────────────────────────────────
MODEL_ZIP  = os.path.join(ROOT, 'models', 'best_model.zip')
VEC_NORM   = os.path.join(ROOT, 'models', 'best_vecnormalize.pkl')
ONNX_OUT   = os.path.join(ROOT, 'models', 'residual_sac_actor.onnx')
NPZ_OUT    = os.path.join(ROOT, 'models', 'vec_normalize_v5_1M.npz')

OBS_DIM    = OBS_DIM_RESIDUAL
ACTION_DIM = 4
OPSET      = 17


# ── Wrapper determinista ──────────────────────────────────────────────────────

class DeterministicActor(nn.Module):
    """
    Wrapper alrededor del actor SB3-SAC que exporta la acción determinista.

    SB3 SAC: actor.get_action_dist_params(obs) → (mean_actions, log_std, kwargs)
    Acción determinista: tanh(mean_actions) ∈ (-1, 1)^4
    """

    def __init__(self, actor):
        super().__init__()
        self.actor = actor

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        mean_actions, _, _ = self.actor.get_action_dist_params(obs)
        return torch.tanh(mean_actions)


# ── Exportar ONNX ─────────────────────────────────────────────────────────────

def export_onnx(model_zip: str = MODEL_ZIP, onnx_out: str = ONNX_OUT) -> None:
    from stable_baselines3 import SAC

    print(f"Cargando modelo SAC desde {model_zip} ...")
    model = SAC.load(model_zip, device='cpu')
    actor = model.policy.actor.eval()

    wrapper = DeterministicActor(actor)
    wrapper.eval()

    dummy_obs = torch.zeros(1, OBS_DIM, dtype=torch.float32)

    print(f"Exportando a ONNX (opset={OPSET}) -> {onnx_out} ...")
    # dynamo=False: usa el exportador TorchScript legacy, compatible con opset 17
    # y sin dependencias de onnxscript ni problemas de encoding en Windows.
    torch.onnx.export(
        wrapper,
        dummy_obs,
        onnx_out,
        opset_version=OPSET,
        input_names=['obs'],
        output_names=['delta'],
        dynamic_axes={
            'obs':   {0: 'batch_size'},
            'delta': {0: 'batch_size'},
        },
        do_constant_folding=True,
        dynamo=False,
    )

    # Verificación rápida
    import onnx
    m = onnx.load(onnx_out)
    onnx.checker.check_model(m)
    print(f"  Modelo ONNX valido. Input: obs[batch,{OBS_DIM}] -> delta[batch,{ACTION_DIM}]")


# ── Volcar VecNormalize ───────────────────────────────────────────────────────

def export_vec_normalize(pkl_path: str = VEC_NORM, npz_out: str = NPZ_OUT) -> None:
    print(f"Leyendo VecNormalize desde {pkl_path} ...")
    with open(pkl_path, 'rb') as f:
        vec_norm = pickle.load(f)

    mean     = np.array(vec_norm.obs_rms.mean,  dtype=np.float32)  # (112,)
    var      = np.array(vec_norm.obs_rms.var,   dtype=np.float32)  # (112,)
    clip_obs = np.float32(vec_norm.clip_obs)                        # escalar

    assert mean.shape == (OBS_DIM,), f"mean.shape={mean.shape}, esperado ({OBS_DIM},)"
    assert var.shape  == (OBS_DIM,), f"var.shape={var.shape}, esperado ({OBS_DIM},)"

    np.savez(npz_out, mean=mean, var=var, clip_obs=clip_obs)
    print(f"  Estadisticas guardadas -> {npz_out}")
    print(f"  clip_obs={clip_obs}  mean[0:3]={mean[:3]}  var[0:3]={var[:3]}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',    default=MODEL_ZIP, help='Ruta al .zip del modelo SAC')
    parser.add_argument('--vec-norm', default=VEC_NORM,  help='Ruta al .pkl de VecNormalize')
    parser.add_argument('--onnx-out', default=ONNX_OUT,  help='Ruta de salida .onnx')
    parser.add_argument('--npz-out',  default=NPZ_OUT,   help='Ruta de salida .npz')
    args = parser.parse_args()

    export_onnx(args.model, args.onnx_out)
    export_vec_normalize(args.vec_norm, args.npz_out)
    print("\nExportación completada.")
    print(f"  {args.onnx_out}")
    print(f"  {args.npz_out}")
