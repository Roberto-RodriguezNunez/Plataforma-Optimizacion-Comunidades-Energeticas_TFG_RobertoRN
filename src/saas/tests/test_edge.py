"""Tests del blueprint edge: POST /api/edge/decision."""
import os
import pytest
from app.models.bateria import Bateria
from app.models.operacion import OperacionHoraria


PAYLOAD_OK = {
    'comunidad_id': None,  # rellenado en cada test con el id real
    'step': 42,
    'consumo_total_kwh': 15.2,
    'gen_total_kwh': 8.5,
    'precio_compra': 0.185,
    'precio_exc': 0.065,
    'soc': 0.72,
    'P_carga_solar': 3.1,
    'P_carga_red': 0.0,
    'P_descarga_casa': 0.0,
    'P_descarga_red': 0.5,
    'beneficio_marginal': 0.32,
}


def _payload(comunidad_id, **overrides):
    p = dict(PAYLOAD_OK, comunidad_id=comunidad_id)
    p.update(overrides)
    return p


class TestEdgeDecision:
    def test_sin_clave_configurada_acepta(self, client, comunidad):
        """Sin EDGE_API_KEY en entorno → cualquier petición se acepta."""
        os.environ.pop('EDGE_API_KEY', None)
        resp = client.post(
            '/api/edge/decision',
            json=_payload(comunidad.__oid__),
        )
        assert resp.status_code == 201
        data = resp.get_json()
        assert data['ok'] is True
        assert 'id' in data

    def test_guarda_operacion_horaria(self, client, comunidad, srp):
        os.environ.pop('EDGE_API_KEY', None)
        assert srp.num_objs(OperacionHoraria) == 0
        client.post('/api/edge/decision', json=_payload(comunidad.__oid__))
        assert srp.num_objs(OperacionHoraria) == 1
        op = OperacionHoraria.query.first()
        assert op.step == 42
        assert abs(op.soc - 0.72) < 1e-6
        assert abs(op.consumo_total_kwh - 15.2) < 1e-6

    def test_actualiza_soc_bateria(self, client, comunidad, bateria, srp):
        os.environ.pop('EDGE_API_KEY', None)
        assert bateria.soc_actual is None
        client.post('/api/edge/decision', json=_payload(comunidad.__oid__, soc=0.55))
        bat = srp.load(Bateria, bateria.__oid__)
        assert bat.soc_actual is not None
        assert abs(bat.soc_actual - 0.55) < 1e-6

    def test_sin_bateria_no_falla(self, client, comunidad, srp):
        """Sin batería registrada, el endpoint igual guarda la operación."""
        os.environ.pop('EDGE_API_KEY', None)
        resp = client.post('/api/edge/decision', json=_payload(comunidad.__oid__))
        assert resp.status_code == 201

    def test_clave_correcta_acepta(self, client, comunidad, monkeypatch):
        monkeypatch.setenv('EDGE_API_KEY', 'secreto123')
        resp = client.post(
            '/api/edge/decision',
            json=_payload(comunidad.__oid__),
            headers={'Authorization': 'Bearer secreto123'},
        )
        assert resp.status_code == 201

    def test_clave_incorrecta_rechaza(self, client, comunidad, monkeypatch):
        monkeypatch.setenv('EDGE_API_KEY', 'secreto123')
        resp = client.post(
            '/api/edge/decision',
            json=_payload(comunidad.__oid__),
            headers={'Authorization': 'Bearer MALA'},
        )
        assert resp.status_code == 401

    def test_sin_header_rechaza_si_clave_configurada(self, client, comunidad, monkeypatch):
        monkeypatch.setenv('EDGE_API_KEY', 'secreto123')
        resp = client.post('/api/edge/decision', json=_payload(comunidad.__oid__))
        assert resp.status_code == 401

    def test_body_vacio_400(self, client):
        os.environ.pop('EDGE_API_KEY', None)
        resp = client.post('/api/edge/decision', data='no-json',
                           content_type='text/plain')
        assert resp.status_code == 400

    def test_campo_faltante_400(self, client, comunidad):
        os.environ.pop('EDGE_API_KEY', None)
        payload = _payload(comunidad.__oid__)
        del payload['soc']
        resp = client.post('/api/edge/decision', json=payload)
        assert resp.status_code == 400
        assert 'soc' in resp.get_json()['error']

    def test_multiples_pasos_acumulan(self, client, comunidad, srp):
        os.environ.pop('EDGE_API_KEY', None)
        for step in range(5):
            client.post('/api/edge/decision',
                        json=_payload(comunidad.__oid__, step=step, soc=0.5 - step * 0.05))
        assert srp.num_objs(OperacionHoraria) == 5
