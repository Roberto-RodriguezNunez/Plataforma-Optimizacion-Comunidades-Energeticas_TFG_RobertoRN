"""Tests de CRUD de viviendas y recálculo de coeficientes."""
import pytest
from app.helpers import oid_to_safe
from app.models.vivienda import Vivienda
from tests.conftest import login_superadmin, login_usuario


class TestListaViviendas:
    def test_lista_global_superadmin(self, client, superadmin, vivienda):
        login_superadmin(client, superadmin)
        resp = client.get('/viviendas/')
        assert resp.status_code == 200

    def test_lista_global_requiere_superadmin(self, client, usuario):
        login_usuario(client, usuario)
        resp = client.get('/viviendas/')
        assert resp.status_code == 403

    def test_lista_comunidad_superadmin(self, client, superadmin, comunidad, vivienda):
        login_superadmin(client, superadmin)
        safe = oid_to_safe(comunidad.__oid__)
        resp = client.get(f'/viviendas/comunidad/{safe}/')
        assert resp.status_code == 200

    def test_lista_comunidad_usuario_con_acceso(self, client, usuario, comunidad, acceso):
        login_usuario(client, usuario)
        safe = oid_to_safe(comunidad.__oid__)
        resp = client.get(f'/viviendas/comunidad/{safe}/')
        assert resp.status_code == 200

    def test_lista_comunidad_usuario_sin_acceso(self, client, usuario, comunidad):
        login_usuario(client, usuario)
        safe = oid_to_safe(comunidad.__oid__)
        resp = client.get(f'/viviendas/comunidad/{safe}/')
        assert resp.status_code == 403


class TestNuevaVivienda:
    def test_crear_ok(self, client, superadmin, comunidad, srp):
        login_superadmin(client, superadmin)
        safe = oid_to_safe(comunidad.__oid__)
        resp = client.post(f'/viviendas/comunidad/{safe}/nueva', data={
            'identificador': 'Piso Nuevo',
            'direccion_completa': 'Calle 1',
            'cups': 'ES9999',
            'potencia_contratada_kw': '4.6',
            'fecha_alta': '2024-01-01',
            'tiene_paneles': '',
        }, follow_redirects=True)
        assert resp.status_code == 200
        v = srp.find_first(Vivienda, lambda v: v.identificador == 'Piso Nuevo')
        assert v is not None

    def test_coeficientes_se_recalculan(self, client, superadmin, comunidad, vivienda, srp):
        """Al crear una segunda vivienda los coeficientes deben sumar 1."""
        login_superadmin(client, superadmin)
        safe = oid_to_safe(comunidad.__oid__)
        client.post(f'/viviendas/comunidad/{safe}/nueva', data={
            'identificador': 'Piso 2B',
            'potencia_contratada_kw': '3.45',
            'fecha_alta': '2024-01-01',
        })
        com_str = str(comunidad.__oid__)
        vivs = [v for v in srp.load_all(Vivienda) if str(v.comunidad_oid) == com_str]
        total = sum(v.coeficiente_reparto for v in vivs)
        assert abs(total - 1.0) < 0.001

    def test_potencia_negativa(self, client, superadmin, comunidad):
        login_superadmin(client, superadmin)
        safe = oid_to_safe(comunidad.__oid__)
        resp = client.post(f'/viviendas/comunidad/{safe}/nueva', data={
            'identificador': 'Negativo',
            'potencia_contratada_kw': '-5',
            'fecha_alta': '2024-01-01',
        })
        assert resp.status_code == 200  # form re-rendered

    def test_potencia_string(self, client, superadmin, comunidad):
        login_superadmin(client, superadmin)
        safe = oid_to_safe(comunidad.__oid__)
        resp = client.post(f'/viviendas/comunidad/{safe}/nueva', data={
            'identificador': 'String',
            'potencia_contratada_kw': 'abc',
            'fecha_alta': '2024-01-01',
        })
        assert resp.status_code == 200

    def test_identificador_vacio(self, client, superadmin, comunidad):
        login_superadmin(client, superadmin)
        safe = oid_to_safe(comunidad.__oid__)
        resp = client.post(f'/viviendas/comunidad/{safe}/nueva', data={
            'identificador': '',
            'potencia_contratada_kw': '3.45',
            'fecha_alta': '2024-01-01',
        })
        assert resp.status_code == 200

    def test_identificador_duplicado(self, client, superadmin, comunidad, vivienda):
        login_superadmin(client, superadmin)
        safe = oid_to_safe(comunidad.__oid__)
        resp = client.post(f'/viviendas/comunidad/{safe}/nueva', data={
            'identificador': 'Piso 1A',
            'potencia_contratada_kw': '3.45',
            'fecha_alta': '2024-01-01',
        }, follow_redirects=True)
        assert b'Ya existe' in resp.data

    def test_requiere_superadmin(self, client, usuario, comunidad):
        login_usuario(client, usuario)
        safe = oid_to_safe(comunidad.__oid__)
        resp = client.get(f'/viviendas/comunidad/{safe}/nueva')
        assert resp.status_code == 403

    def test_con_paneles(self, client, superadmin, comunidad, srp):
        login_superadmin(client, superadmin)
        safe = oid_to_safe(comunidad.__oid__)
        resp = client.post(f'/viviendas/comunidad/{safe}/nueva', data={
            'identificador': 'Solar',
            'potencia_contratada_kw': '5.0',
            'fecha_alta': '2024-01-01',
            'tiene_paneles': 'y',
            'potencia_pico_paneles_kwp': '3.2',
            'numero_paneles': '8',
            'fecha_instalacion_paneles': '2024-06-01',
            'orientacion_paneles': 'sur',
        }, follow_redirects=True)
        assert resp.status_code == 200
        v = srp.find_first(Vivienda, lambda v: v.identificador == 'Solar')
        assert v.tiene_paneles is True
        assert v.numero_paneles == 8

    def test_paneles_negativo(self, client, superadmin, comunidad):
        login_superadmin(client, superadmin)
        safe = oid_to_safe(comunidad.__oid__)
        resp = client.post(f'/viviendas/comunidad/{safe}/nueva', data={
            'identificador': 'PanelNeg',
            'potencia_contratada_kw': '3.0',
            'fecha_alta': '2024-01-01',
            'tiene_paneles': 'y',
            'numero_paneles': '-5',
        })
        assert resp.status_code == 200


class TestDetalleVivienda:
    def test_detalle_superadmin(self, client, superadmin, vivienda):
        login_superadmin(client, superadmin)
        safe = oid_to_safe(vivienda.__oid__)
        resp = client.get(f'/viviendas/{safe}')
        assert resp.status_code == 200

    def test_detalle_usuario_con_acceso(self, client, usuario, vivienda, acceso):
        login_usuario(client, usuario)
        safe = oid_to_safe(vivienda.__oid__)
        resp = client.get(f'/viviendas/{safe}')
        assert resp.status_code == 200

    def test_detalle_usuario_sin_acceso(self, client, usuario, vivienda):
        login_usuario(client, usuario)
        safe = oid_to_safe(vivienda.__oid__)
        resp = client.get(f'/viviendas/{safe}')
        assert resp.status_code == 403

    def test_detalle_oid_invalido(self, client, superadmin):
        login_superadmin(client, superadmin)
        resp = client.get('/viviendas/oid-falso')
        assert resp.status_code == 404


class TestEditarVivienda:
    def test_editar_potencia_recalcula(self, client, superadmin, comunidad, vivienda, srp):
        login_superadmin(client, superadmin)
        # Crear segunda vivienda
        safe_com = oid_to_safe(comunidad.__oid__)
        client.post(f'/viviendas/comunidad/{safe_com}/nueva', data={
            'identificador': 'Piso 2',
            'potencia_contratada_kw': '3.45',
            'fecha_alta': '2024-01-01',
        })
        # Editar primera cambiando potencia
        safe = oid_to_safe(vivienda.__oid__)
        client.post(f'/viviendas/{safe}/editar', data={
            'identificador': 'Piso 1A',
            'potencia_contratada_kw': '6.9',
            'fecha_alta': '2024-01-01',
        }, follow_redirects=True)
        com_str = str(comunidad.__oid__)
        vivs = [v for v in srp.load_all(Vivienda) if str(v.comunidad_oid) == com_str]
        total = sum(v.coeficiente_reparto for v in vivs)
        assert abs(total - 1.0) < 0.001


class TestEliminarVivienda:
    def test_eliminar_recalcula(self, client, superadmin, comunidad, srp):
        """Al eliminar una vivienda los coeficientes restantes deben sumar 1."""
        login_superadmin(client, superadmin)
        safe_com = oid_to_safe(comunidad.__oid__)
        # Crear 3 viviendas
        for i in range(3):
            client.post(f'/viviendas/comunidad/{safe_com}/nueva', data={
                'identificador': f'V{i}',
                'potencia_contratada_kw': '3.0',
                'fecha_alta': '2024-01-01',
            })
        com_str = str(comunidad.__oid__)
        vivs = [v for v in srp.load_all(Vivienda) if str(v.comunidad_oid) == com_str]
        # Eliminar una
        safe_del = oid_to_safe(vivs[0].__oid__)
        client.post(f'/viviendas/{safe_del}/eliminar')
        vivs_after = [v for v in srp.load_all(Vivienda) if str(v.comunidad_oid) == com_str]
        total = sum(v.coeficiente_reparto for v in vivs_after)
        assert abs(total - 1.0) < 0.001

    def test_eliminar_requiere_superadmin(self, client, usuario, vivienda):
        login_usuario(client, usuario)
        safe = oid_to_safe(vivienda.__oid__)
        resp = client.post(f'/viviendas/{safe}/eliminar')
        assert resp.status_code == 403
