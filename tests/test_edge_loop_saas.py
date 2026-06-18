"""Tests del comportamiento del edge_loop cuando el SaaS no está disponible.

Verifican que:
- El bucle continúa aunque el SaaS rechace las conexiones.
- El wait loop desactiva _post_saas si el SaaS nunca responde.
- Los posts al SaaS son silenciosos (no rompen el bucle).
"""
from __future__ import annotations

import json
import os
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip('numpy', reason='numpy no disponible — ejecutar dentro del contenedor edge')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_requests_stub(health_ok: bool, post_raises: bool = False):
    """Devuelve un módulo requests simulado."""
    stub = types.ModuleType('requests')

    def _get(url, timeout=5):
        if health_ok:
            r = MagicMock()
            r.status_code = 200
            return r
        raise ConnectionRefusedError('SaaS caído')

    def _post(url, json=None, headers=None, timeout=5):
        if post_raises:
            raise ConnectionRefusedError('SaaS caído')
        r = MagicMock()
        r.status_code = 201
        r.content = b'{"ok": true}'
        r.json.return_value = {'ok': True}
        return r

    stub.get  = _get
    stub.post = _post
    return stub


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEdgeLoopSaasDesconectado:
    """El bucle corre aunque el SaaS no esté disponible."""

    def test_bucle_completa_sin_saas(self):
        """Con SAAS_API_URL pero SaaS caído, el bucle completa N_STEPS sin excepción."""
        requests_stub = _make_requests_stub(health_ok=False)

        env = {
            'FEED_MODE': 'simulado',
            'N_STEPS': '5',
            'SAAS_API_URL': 'http://fake-saas:5000',
            'EDGE_API_KEY': 'test-key',
            'COMUNIDAD_ID': '1',
        }

        with patch.dict(os.environ, env):
            with patch.dict(sys.modules, {'requests': requests_stub}):
                # Reimportar el módulo para que tome las env vars y el stub
                if 'src.production.edge_loop' in sys.modules:
                    del sys.modules['src.production.edge_loop']

                from src.production import edge_loop
                # Parchear _time.sleep para que el wait loop no espere de verdad
                with patch('time.sleep', return_value=None):
                    # No debe lanzar excepción
                    edge_loop.run(n_steps=5)

    def test_post_fallido_no_rompe_bucle(self):
        """Aunque el POST al SaaS falle en cada step, el bucle completa todos los pasos."""
        requests_stub = _make_requests_stub(health_ok=True, post_raises=True)

        env = {
            'FEED_MODE': 'simulado',
            'N_STEPS': '5',
            'SAAS_API_URL': 'http://fake-saas:5000',
            'EDGE_API_KEY': 'test-key',
            'COMUNIDAD_ID': '1',
        }

        outputs = []
        original_print = print

        def capture_print(*args, **kwargs):
            if args and isinstance(args[0], str):
                outputs.append(args[0])
            original_print(*args, **kwargs)

        with patch.dict(os.environ, env):
            with patch.dict(sys.modules, {'requests': requests_stub}):
                if 'src.production.edge_loop' in sys.modules:
                    del sys.modules['src.production.edge_loop']

                from src.production import edge_loop
                with patch('builtins.print', side_effect=capture_print):
                    edge_loop.run(n_steps=5)

        step_lines = [o for o in outputs if '"step"' in o]
        assert len(step_lines) == 5, f'Se esperaban 5 steps, se obtuvieron {len(step_lines)}'

        done_lines = [o for o in outputs if '"done"' in o]
        assert len(done_lines) == 1

    def test_bucle_emite_evento_done(self):
        """El último JSON emitido siempre contiene event=done."""
        requests_stub = _make_requests_stub(health_ok=False)

        env = {
            'FEED_MODE': 'simulado',
            'N_STEPS': '3',
            'SAAS_API_URL': '',  # sin SaaS → sin intentos de conexión
        }

        outputs = []

        with patch.dict(os.environ, env):
            with patch.dict(sys.modules, {'requests': requests_stub}):
                if 'src.production.edge_loop' in sys.modules:
                    del sys.modules['src.production.edge_loop']

                from src.production import edge_loop
                with patch('builtins.print', side_effect=lambda *a, **kw: outputs.append(a[0] if a else '')):
                    edge_loop.run(n_steps=3)

        last_json = json.loads(outputs[-1])
        assert last_json['event'] == 'done'
        assert last_json['n_steps'] == 3
