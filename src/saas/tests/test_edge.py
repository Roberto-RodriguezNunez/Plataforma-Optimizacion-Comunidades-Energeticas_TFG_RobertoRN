"""Tests del blueprint edge: POST /api/edge/decision y /api/edge/generar-cierre."""
import os
from datetime import datetime, timezone
import pytest
from app.extensions import db
from app.models.bateria import Bateria
from app.models.cierre import CierreMensual
from app.models.operacion import OperacionHoraria
from app.models.vivienda import Vivienda


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


class TestGenerarCierreEdge:
    """Tests de POST /api/edge/generar-cierre."""

    def _insertar_op(self, comunidad_oid, mes='2025-01'):
        year, month = int(mes[:4]), int(mes[5:])
        op = OperacionHoraria(
            comunidad_oid=comunidad_oid,
            ts=datetime(year, month, 15, 12, tzinfo=timezone.utc),
            step=0,
            consumo_total_kwh=10.0, gen_total_kwh=4.0,
            precio_compra=0.20, precio_exc=0.06, soc=0.6,
            p_carga_solar=2.0, p_carga_red=0.0,
            p_descarga_casa=1.5, p_descarga_red=0.0,
            beneficio_marginal=0.15,
        )
        db.session.add(op)

    def _vivienda(self, comunidad_oid):
        v = Vivienda(
            comunidad_oid=comunidad_oid, identificador='V1', cups='ES9999',
            potencia_contratada_kw=4.6, coeficiente_reparto=1.0,
            tiene_paneles=True, potencia_pico_paneles_kwp=4.0,
            numero_paneles=10, orientacion_paneles='sur',
        )
        db.session.add(v)
        db.session.commit()

    def test_generar_cierre_crea_cierre_mensual(self, client, comunidad, srp):
        os.environ.pop('EDGE_API_KEY', None)
        self._vivienda(comunidad.__oid__)
        self._insertar_op(comunidad.__oid__, '2025-01')
        db.session.commit()

        resp = client.post('/api/edge/generar-cierre',
                           json={'comunidad_id': comunidad.__oid__, 'mes': '2025-01'})
        assert resp.status_code == 201
        data = resp.get_json()
        assert data['ok'] is True
        assert data['n_viviendas'] == 1
        assert srp.num_objs(CierreMensual) == 1

    def test_generar_cierre_sin_operaciones_400(self, client, comunidad, srp):
        os.environ.pop('EDGE_API_KEY', None)
        self._vivienda(comunidad.__oid__)
        resp = client.post('/api/edge/generar-cierre',
                           json={'comunidad_id': comunidad.__oid__, 'mes': '2025-06'})
        assert resp.status_code == 400
        assert srp.num_objs(CierreMensual) == 0

    def test_generar_cierre_requiere_auth(self, client, comunidad, monkeypatch):
        monkeypatch.setenv('EDGE_API_KEY', 'secreto')
        resp = client.post('/api/edge/generar-cierre',
                           json={'comunidad_id': comunidad.__oid__, 'mes': '2025-01'})
        assert resp.status_code == 401

    def test_generar_cierre_campos_requeridos(self, client):
        os.environ.pop('EDGE_API_KEY', None)
        resp = client.post('/api/edge/generar-cierre', json={'mes': '2025-01'})
        assert resp.status_code == 400
