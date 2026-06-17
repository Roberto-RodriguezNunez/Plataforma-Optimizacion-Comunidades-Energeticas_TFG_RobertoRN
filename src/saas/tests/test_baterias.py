"""Tests de gestión de baterías."""
import pytest
from app.helpers import oid_to_safe
from app.models.bateria import Bateria
from tests.conftest import login_superadmin, login_usuario


class TestBaterias:
    def test_detalle_comunidad(self, client, superadmin, comunidad, bateria):
        login_superadmin(client, superadmin)
        safe = oid_to_safe(comunidad.__oid__)
        resp = client.get(f'/baterias/comunidad/{safe}/')
        assert resp.status_code == 200

    def test_detalle_requiere_superadmin(self, client, usuario, comunidad):
        login_usuario(client, usuario)
        safe = oid_to_safe(comunidad.__oid__)
        resp = client.get(f'/baterias/comunidad/{safe}/')
        assert resp.status_code == 403

    def test_crear_ok(self, client, superadmin, comunidad, srp):
        login_superadmin(client, superadmin)
        safe = oid_to_safe(comunidad.__oid__)
        resp = client.post(f'/baterias/comunidad/{safe}/nueva', data={
            'capacidad_nominal_kwh': '25.0',
            'capacidad_util_actual_kwh': '24.0',
            'ciclos_acumulados': '50',
            'fecha_instalacion': '2024-01-01',
            'fabricante': 'Tesla',
            'modelo': 'Powerwall',
            'estado': 'operativa',
        }, follow_redirects=True)
        assert resp.status_code == 200

    def test_crear_duplicada(self, client, superadmin, comunidad, bateria):
        """Solo una batería por comunidad."""
        login_superadmin(client, superadmin)
        safe = oid_to_safe(comunidad.__oid__)
        resp = client.post(f'/baterias/comunidad/{safe}/nueva', data={
            'capacidad_nominal_kwh': '10',
            'capacidad_util_actual_kwh': '9',
            'ciclos_acumulados': '10',
            'fecha_instalacion': '2024-01-01',
            'estado': 'operativa',
        }, follow_redirects=True)
        assert b'ya tiene' in resp.data

    def test_capacidad_negativa(self, client, superadmin, comunidad):
        login_superadmin(client, superadmin)
        safe = oid_to_safe(comunidad.__oid__)
        resp = client.post(f'/baterias/comunidad/{safe}/nueva', data={
            'capacidad_nominal_kwh': '-5',
            'capacidad_util_actual_kwh': '0',
            'ciclos_acumulados': '0',
            'fecha_instalacion': '2024-01-01',
            'estado': 'operativa',
        })
        assert resp.status_code == 200  # form re-rendered

    def test_capacidad_string(self, client, superadmin, comunidad):
        login_superadmin(client, superadmin)
        safe = oid_to_safe(comunidad.__oid__)
        resp = client.post(f'/baterias/comunidad/{safe}/nueva', data={
            'capacidad_nominal_kwh': 'abc',
            'capacidad_util_actual_kwh': '0',
            'ciclos_acumulados': '0',
            'fecha_instalacion': '2024-01-01',
            'estado': 'operativa',
        })
        assert resp.status_code == 200

    def test_ciclos_negativos(self, client, superadmin, comunidad):
        login_superadmin(client, superadmin)
        safe = oid_to_safe(comunidad.__oid__)
        resp = client.post(f'/baterias/comunidad/{safe}/nueva', data={
            'capacidad_nominal_kwh': '10',
            'capacidad_util_actual_kwh': '9',
            'ciclos_acumulados': '-100',
            'fecha_instalacion': '2024-01-01',
            'estado': 'operativa',
        })
        assert resp.status_code == 200

    def test_estado_invalido(self, client, superadmin, comunidad):
        login_superadmin(client, superadmin)
        safe = oid_to_safe(comunidad.__oid__)
        resp = client.post(f'/baterias/comunidad/{safe}/nueva', data={
            'capacidad_nominal_kwh': '10',
            'capacidad_util_actual_kwh': '9',
            'ciclos_acumulados': '0',
            'fecha_instalacion': '2024-01-01',
            'estado': 'inventado',
        })
        assert resp.status_code == 200

    def test_cambiar_estado(self, client, superadmin, bateria):
        login_superadmin(client, superadmin)
        safe = oid_to_safe(bateria.__oid__)
        resp = client.post(f'/baterias/{safe}/cambiar-estado', data={
            'estado': 'mantenimiento',
        }, follow_redirects=True)
        assert resp.status_code == 200

    def test_eliminar(self, client, superadmin, bateria, srp):
        login_superadmin(client, superadmin)
        safe = oid_to_safe(bateria.__oid__)
        resp = client.post(f'/baterias/{safe}/eliminar', follow_redirects=True)
        assert resp.status_code == 200
        assert srp.num_objs(Bateria) == 0
