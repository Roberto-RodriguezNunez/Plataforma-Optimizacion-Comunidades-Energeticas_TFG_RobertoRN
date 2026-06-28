"""
registro.py — Loader del registry de experimentos (config/experimentos.yaml)
============================================================================
Fuente única de hiperparámetros de cada versión entrenable. Resuelve
`base ⊕ override` y expone:

    cargar_version(familia, version) -> dict   (hiperparámetros resueltos + meta)
    listar_versiones(familia)        -> [str]
    familias()                       -> [str]
    seeds_comunes()                  -> (train_seeds, eval_seeds, eval_episodes)
    seeds_para_version(familia, ver)  -> train_seeds recortado a n_seeds de la versión

El merge es a nivel de clave de primer nivel: un override que fija `learning_rate`
(escalar o {init,final}) reemplaza por completo el valor del base — justo lo que se
quiere para alternar entre constante y decay.
"""

import os
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_REGISTRY_PATH = os.path.join(ROOT, "config", "experimentos.yaml")

with open(_REGISTRY_PATH, "r", encoding="utf-8") as _f:
    _REG = yaml.safe_load(_f)


def familias():
    """Lista de familias (todas las claves salvo 'comun')."""
    return [k for k in _REG.keys() if k != "comun"]


def _familia(familia: str) -> dict:
    if familia not in _REG:
        raise KeyError(
            f"Familia '{familia}' no existe en experimentos.yaml. "
            f"Disponibles: {familias()}"
        )
    return _REG[familia]


def listar_versiones(familia: str):
    """Nombres de versión de una familia, p.ej. ['DQN-1', ..., 'DQN-6']."""
    return list(_familia(familia)["versiones"].keys())


def cargar_version(familia: str, version: str) -> dict:
    """Devuelve los hiperparámetros resueltos (base ⊕ override) de una versión.

    El dict incluye además las claves meta: 'algo', 'env', 'familia', 'version',
    'nota'. Lanza KeyError si la familia o la versión no existen.
    """
    fam = _familia(familia)
    versiones = fam["versiones"]
    if version not in versiones:
        raise KeyError(
            f"Versión '{version}' no existe en la familia '{familia}'. "
            f"Disponibles: {listar_versiones(familia)}"
        )

    cfg = dict(fam.get("base", {}))          # copia del base
    cfg.update(versiones[version] or {})      # override de primer nivel

    cfg["algo"] = fam["algo"]
    cfg["env"] = fam["env"]
    cfg["familia"] = familia
    cfg["version"] = version
    cfg.setdefault("nota", "")
    return cfg


def seeds_comunes():
    """(train_seeds, eval_seeds, eval_episodes) comunes a toda la suite."""
    c = _REG["comun"]
    return list(c["train_seeds"]), int(c["eval_seeds"]), int(c["eval_episodes"])


def seeds_para_version(familia: str, version: str):
    """train_seeds recortado a las `n_seeds` que pida la versión.

    Devuelve los primeros N de `comun.train_seeds` (determinista). Si la versión
    no declara `n_seeds`, usa todas (comportamiento por defecto: 3 semillas).
    Pensado para entrenar 1 semilla en barridos/ablaciones y 3 en candidatas.
    """
    train_seeds, _, _ = seeds_comunes()
    n = int(cargar_version(familia, version).get("n_seeds", len(train_seeds)))
    n = max(1, min(n, len(train_seeds)))
    return train_seeds[:n]


def es_decay(valor) -> bool:
    """True si un hiperparámetro está expresado como decay {init, final}."""
    return isinstance(valor, dict) and "init" in valor and "final" in valor


def schedule_lineal(valor):
    """Convierte un hiperparámetro a callable de SB3 (progress_remaining→valor).

    - escalar  -> función constante
    - {init,final} -> decay lineal (progress_remaining va de 1.0 a 0.0)
    """
    if es_decay(valor):
        ini, fin = float(valor["init"]), float(valor["final"])
        return lambda p: fin + p * (ini - fin)
    v = float(valor)
    return lambda p: v
