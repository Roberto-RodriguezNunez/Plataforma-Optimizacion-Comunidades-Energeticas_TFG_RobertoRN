"""Tests de CRUD de comunidades."""
import pytest
from app.helpers import oid_to_safe
from app.models.comunidad import Comunidad
from tests.conftest import login_superadmin, login_usuario


class TestListaComunidades:
    def test_requiere_superadmin(self, client, usuario):
        login_usuario(client, usuario)
        resp = client.get('/comunidades/')
        assert resp.status_code == 403

    def test_lista_ok(self, client, superadmin, comunidad):
        login_superadmin(client, superadmin)
        resp = client.get('/comunidades/')
        assert resp.status_code == 200
        assert b'Comunidad Test' in resp.data

    def test_busqueda(self, client, superadmin, comunidad):
        login_superadmin(client, superadmin)
        resp = client.get('/comunidades/?q=vigo')
        assert resp.status_code == 200

    def test_filtro_estado(self, client, superadmin, comunidad):
        login_superadmin(client, superadmin)
        resp = client.get('/comunidades/?estado=activa')
        assert resp.status_code == 200


class TestNuevaComunidad:
    def test_requiere_superadmin(self, client, usuario):
        login_usuario(client, usuario)
        resp = client.get('/comunidades/nueva')
        assert resp.status_code == 403

    def test_crear_ok(self, client, superadmin, srp):
        login_superadmin(client, superadmin)
        resp = client.post('/comunidades/nueva', data={
            'nombre': 'Nueva Comunidad',
            'ubicacion': 'Santiago',
            'fecha_constitucion': '2024-06-01',
            'estado': 'activa',
            'descripcion': 'Test',
        }, follow_redirects=True)
        assert resp.status_code == 200
        c = srp.find_first(Comunidad, lambda c: c.nombre == 'Nueva Comunidad')
        assert c is not None

    def test_nombre_duplicado(self, client, superadmin, comunidad):
        login_superadmin(client, superadmin)
        resp = client.post('/comunidades/nueva', data={
            'nombre': 'Comunidad Test',
            'ubicacion': 'Santiago',
            'fecha_constitucion': '2024-06-01',
            'estado': 'activa',
        }, follow_redirects=True)
        assert b'Ya existe' in resp.data

    def test_nombre_vacio(self, client, superadmin):
        login_superadmin(client, superadmin)
        resp = client.post('/comunidades/nueva', data={
            'nombre': '',
            'ubicacion': 'Santiago',
            'fecha_constitucion': '2024-06-01',
            'estado': 'activa',
        })
        assert resp.status_code == 200  # form re-rendered

    def test_nombre_muy_corto(self, client, superadmin):
        login_superadmin(client, superadmin)
        resp = client.post('/comunidades/nueva', data={
            'nombre': 'A',
            'ubicacion': 'Santiago',
            'fecha_constitucion': '2024-06-01',
            'estado': 'activa',
        })
        assert resp.status_code == 200

    def test_estado_invalido(self, client, superadmin):
        login_superadmin(client, superadmin)
        resp = client.post('/comunidades/nueva', data={
            'nombre': 'Estado Mal',
            'ubicacion': 'Lugar',
            'fecha_constitucion': '2024-06-01',
            'estado': 'inventado',
        })
        assert resp.status_code == 200

    def test_fecha_invalida(self, client, superadmin):
        login_superadmin(client, superadmin)
        resp = client.post('/comunidades/nueva', data={
            'nombre': 'Fecha Mal',
            'ubicacion': 'Lugar',
            'fecha_constitucion': 'no-es-fecha',
            'estado': 'activa',
        })
        assert resp.status_code == 200


class TestDetalleComunidad:
    def test_detalle_superadmin(self, client, superadmin, comunidad):
        login_superadmin(client, superadmin)
        safe = oid_to_safe(comunidad.__oid__)
        resp = client.get(f'/comunidades/{safe}')
        assert resp.status_code == 200
        assert b'Comunidad Test' in resp.data

    def test_detalle_usuario_sin_acceso(self, client, usuario, comunidad):
        login_usuario(client, usuario)
        safe = oid_to_safe(comunidad.__oid__)
        resp = client.get(f'/comunidades/{safe}')
        assert resp.status_code == 403

    def test_detalle_usuario_con_acceso(self, client, usuario, comunidad, acceso):
        login_usuario(client, usuario)
        safe = oid_to_safe(comunidad.__oid__)
        resp = client.get(f'/comunidades/{safe}')
        assert resp.status_code == 200

    def test_detalle_oid_invalido(self, client, superadmin):
        login_superadmin(client, superadmin)
        resp = client.get('/comunidades/oid-inexistente')
        assert resp.status_code == 404


class TestEditarComunidad:
    def test_editar_ok(self, client, superadmin, comunidad, srp):
        login_superadmin(client, superadmin)
        safe = oid_to_safe(comunidad.__oid__)
        resp = client.post(f'/comunidades/{safe}/editar', data={
            'nombre': 'Editada',
            'ubicacion': 'Madrid',
            'fecha_constitucion': '2024-01-01',
            'estado': 'activa',
        }, follow_redirects=True)
        assert resp.status_code == 200

    def test_editar_requiere_superadmin(self, client, usuario, comunidad):
        login_usuario(client, usuario)
        safe = oid_to_safe(comunidad.__oid__)
        resp = client.get(f'/comunidades/{safe}/editar')
        assert resp.status_code == 403


class TestEliminarComunidad:
    def test_eliminar_ok(self, client, superadmin, comunidad, srp):
        login_superadmin(client, superadmin)
        safe = oid_to_safe(comunidad.__oid__)
        resp = client.post(f'/comunidades/{safe}/eliminar', follow_redirects=True)
        assert resp.status_code == 200

    def test_eliminar_requiere_superadmin(self, client, usuario, comunidad):
        login_usuario(client, usuario)
        safe = oid_to_safe(comunidad.__oid__)
        resp = client.post(f'/comunidades/{safe}/eliminar')
        assert resp.status_code == 403

    def test_eliminar_ajax(self, client, superadmin, comunidad):
        login_superadmin(client, superadmin)
        safe = oid_to_safe(comunidad.__oid__)
        resp = client.post(f'/comunidades/{safe}/eliminar',
                           headers={'X-Requested-With': 'XMLHttpRequest'})
        assert resp.status_code == 200
        assert resp.json['success'] is True

    def test_eliminar_cascada(self, client, superadmin, comunidad, vivienda, acceso, srp):
        """Eliminar comunidad debe borrar viviendas y accesos."""
        login_superadmin(client, superadmin)
        safe = oid_to_safe(comunidad.__oid__)
        client.post(f'/comunidades/{safe}/eliminar')
        from app.models.vivienda import Vivienda
        from app.models.acceso import AccesoVivienda
        assert srp.num_objs(Vivienda) == 0
        assert srp.num_objs(AccesoVivienda) == 0
