"""Tests de cierres mensuales."""
import pytest
from app.helpers import oid_to_safe
from app.models.cierre import CierreMensual
from tests.conftest import login_superadmin, login_usuario


class TestListaCierres:
    def test_lista_superadmin(self, client, superadmin, vivienda, cierre):
        login_superadmin(client, superadmin)
        safe = oid_to_safe(vivienda.__oid__)
        resp = client.get(f'/cierres/vivienda/{safe}/')
        assert resp.status_code == 200

    def test_lista_usuario_con_acceso(self, client, usuario, vivienda, acceso, cierre):
        login_usuario(client, usuario)
        safe = oid_to_safe(vivienda.__oid__)
        resp = client.get(f'/cierres/vivienda/{safe}/')
        assert resp.status_code == 200

    def test_lista_usuario_sin_acceso(self, client, usuario, vivienda, cierre):
        login_usuario(client, usuario)
        safe = oid_to_safe(vivienda.__oid__)
        resp = client.get(f'/cierres/vivienda/{safe}/')
        assert resp.status_code == 403


class TestNuevoCierre:
    def test_crear_ok(self, client, superadmin, vivienda, srp):
        login_superadmin(client, superadmin)
        safe = oid_to_safe(vivienda.__oid__)
        resp = client.post(f'/cierres/vivienda/{safe}/nuevo', data={
            'mes': '2024-11',
            'consumo_total_kwh': '280',
            'autoconsumo_directo_kwh': '70',
            'energia_de_bateria_kwh': '35',
            'vertido_a_red_kwh': '8',
            'ahorro_eur': '20',
            'factura_escenario_base_eur': '61.6',
            'factura_escenario_real_eur': '41.6',
            'porcentaje_ahorro_global': '5',
            'coeficiente_reparto_aplicado': '0.5',
        }, follow_redirects=True)
        assert resp.status_code == 200
        assert srp.num_objs(CierreMensual) == 1

    def test_mes_duplicado(self, client, superadmin, vivienda, cierre):
        login_superadmin(client, superadmin)
        safe = oid_to_safe(vivienda.__oid__)
        resp = client.post(f'/cierres/vivienda/{safe}/nuevo', data={
            'mes': '2024-10',  # ya existe
            'consumo_total_kwh': '100',
            'autoconsumo_directo_kwh': '20',
            'energia_de_bateria_kwh': '10',
            'vertido_a_red_kwh': '0',
            'ahorro_eur': '5',
            'factura_escenario_base_eur': '22',
            'factura_escenario_real_eur': '17',
            'porcentaje_ahorro_global': '2',
            'coeficiente_reparto_aplicado': '0.5',
        }, follow_redirects=True)
        assert b'Ya existe' in resp.data

    def test_mes_formato_invalido(self, client, superadmin, vivienda):
        login_superadmin(client, superadmin)
        safe = oid_to_safe(vivienda.__oid__)
        resp = client.post(f'/cierres/vivienda/{safe}/nuevo', data={
            'mes': 'octubre',
            'consumo_total_kwh': '100',
            'autoconsumo_directo_kwh': '20',
            'energia_de_bateria_kwh': '10',
            'vertido_a_red_kwh': '0',
            'ahorro_eur': '5',
            'factura_escenario_base_eur': '22',
            'factura_escenario_real_eur': '17',
            'porcentaje_ahorro_global': '2',
            'coeficiente_reparto_aplicado': '0.5',
        })
        assert resp.status_code == 200  # form re-rendered

    def test_consumo_negativo(self, client, superadmin, vivienda):
        login_superadmin(client, superadmin)
        safe = oid_to_safe(vivienda.__oid__)
        resp = client.post(f'/cierres/vivienda/{safe}/nuevo', data={
            'mes': '2024-11',
            'consumo_total_kwh': '-100',
            'autoconsumo_directo_kwh': '20',
            'energia_de_bateria_kwh': '10',
            'vertido_a_red_kwh': '0',
            'ahorro_eur': '5',
            'factura_escenario_base_eur': '22',
            'factura_escenario_real_eur': '17',
            'porcentaje_ahorro_global': '2',
            'coeficiente_reparto_aplicado': '0.5',
        })
        assert resp.status_code == 200

    def test_consumo_string(self, client, superadmin, vivienda):
        login_superadmin(client, superadmin)
        safe = oid_to_safe(vivienda.__oid__)
        resp = client.post(f'/cierres/vivienda/{safe}/nuevo', data={
            'mes': '2024-11',
            'consumo_total_kwh': 'abc',
            'autoconsumo_directo_kwh': '20',
            'energia_de_bateria_kwh': '10',
            'vertido_a_red_kwh': '0',
            'ahorro_eur': '5',
            'factura_escenario_base_eur': '22',
            'factura_escenario_real_eur': '17',
            'porcentaje_ahorro_global': '2',
            'coeficiente_reparto_aplicado': '0.5',
        })
        assert resp.status_code == 200

    def test_porcentaje_fuera_rango(self, client, superadmin, vivienda):
        login_superadmin(client, superadmin)
        safe = oid_to_safe(vivienda.__oid__)
        resp = client.post(f'/cierres/vivienda/{safe}/nuevo', data={
            'mes': '2024-11',
            'consumo_total_kwh': '100',
            'autoconsumo_directo_kwh': '20',
            'energia_de_bateria_kwh': '10',
            'vertido_a_red_kwh': '0',
            'ahorro_eur': '5',
            'factura_escenario_base_eur': '22',
            'factura_escenario_real_eur': '17',
            'porcentaje_ahorro_global': '150',  # > 100
            'coeficiente_reparto_aplicado': '0.5',
        })
        assert resp.status_code == 200

    def test_coeficiente_fuera_rango(self, client, superadmin, vivienda):
        login_superadmin(client, superadmin)
        safe = oid_to_safe(vivienda.__oid__)
        resp = client.post(f'/cierres/vivienda/{safe}/nuevo', data={
            'mes': '2024-11',
            'consumo_total_kwh': '100',
            'autoconsumo_directo_kwh': '20',
            'energia_de_bateria_kwh': '10',
            'vertido_a_red_kwh': '0',
            'ahorro_eur': '5',
            'factura_escenario_base_eur': '22',
            'factura_escenario_real_eur': '17',
            'porcentaje_ahorro_global': '5',
            'coeficiente_reparto_aplicado': '2.5',  # > 1
        })
        assert resp.status_code == 200

    def test_requiere_superadmin(self, client, usuario, vivienda):
        login_usuario(client, usuario)
        safe = oid_to_safe(vivienda.__oid__)
        resp = client.get(f'/cierres/vivienda/{safe}/nuevo')
        assert resp.status_code == 403


class TestDetalleCierre:
    def test_detalle_superadmin(self, client, superadmin, cierre):
        login_superadmin(client, superadmin)
        safe = oid_to_safe(cierre.__oid__)
        resp = client.get(f'/cierres/{safe}')
        assert resp.status_code == 200

    def test_detalle_usuario_con_acceso(self, client, usuario, cierre, acceso):
        login_usuario(client, usuario)
        safe = oid_to_safe(cierre.__oid__)
        resp = client.get(f'/cierres/{safe}')
        assert resp.status_code == 200


class TestEliminarCierre:
    def test_eliminar_ok(self, client, superadmin, cierre, srp):
        login_superadmin(client, superadmin)
        safe = oid_to_safe(cierre.__oid__)
        resp = client.post(f'/cierres/{safe}/eliminar', follow_redirects=True)
        assert resp.status_code == 200
        assert srp.num_objs(CierreMensual) == 0

    def test_eliminar_requiere_superadmin(self, client, usuario, cierre, acceso):
        login_usuario(client, usuario)
        safe = oid_to_safe(cierre.__oid__)
        resp = client.post(f'/cierres/{safe}/eliminar')
        assert resp.status_code == 403
