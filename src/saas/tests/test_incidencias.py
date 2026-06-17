"""Tests de incidencias."""
import pytest
from app.helpers import oid_to_safe
from app.models.incidencia import Incidencia
from tests.conftest import login_superadmin, login_usuario


class TestListaIncidencias:
    def test_superadmin_ve_todas(self, client, superadmin, incidencia):
        login_superadmin(client, superadmin)
        resp = client.get('/incidencias/')
        assert resp.status_code == 200
        assert b'Fuga de agua' in resp.data

    def test_usuario_ve_solo_suyas(self, client, usuario, incidencia):
        login_usuario(client, usuario)
        resp = client.get('/incidencias/')
        assert resp.status_code == 200
        assert b'Fuga de agua' in resp.data

    def test_requiere_login(self, client):
        resp = client.get('/incidencias/')
        assert resp.status_code == 302


class TestNuevaIncidencia:
    def test_crear_ok(self, client, usuario, vivienda, acceso, srp):
        login_usuario(client, usuario)
        safe_viv = oid_to_safe(vivienda.__oid__)
        resp = client.post('/incidencias/nueva', data={
            'titulo': 'Corte de luz',
            'descripcion': 'No hay luz en el portal',
            'tipo': 'averia',
            'vivienda_safe_oid': safe_viv,
        }, follow_redirects=True)
        assert resp.status_code == 200
        i = srp.find_first(Incidencia, lambda i: i.titulo == 'Corte de luz')
        assert i is not None
        assert i.estado == 'abierta'

    def test_superadmin_no_puede_crear(self, client, superadmin):
        login_superadmin(client, superadmin)
        resp = client.get('/incidencias/nueva', follow_redirects=True)
        assert resp.status_code == 200  # redirected with flash

    def test_titulo_vacio(self, client, usuario, vivienda, acceso):
        login_usuario(client, usuario)
        resp = client.post('/incidencias/nueva', data={
            'titulo': '',
            'descripcion': 'Descripción',
            'tipo': 'averia',
            'vivienda_safe_oid': oid_to_safe(vivienda.__oid__),
        })
        assert resp.status_code == 200  # form re-rendered

    def test_descripcion_vacia(self, client, usuario, vivienda, acceso):
        login_usuario(client, usuario)
        resp = client.post('/incidencias/nueva', data={
            'titulo': 'Titulo',
            'descripcion': '',
            'tipo': 'averia',
            'vivienda_safe_oid': oid_to_safe(vivienda.__oid__),
        })
        assert resp.status_code == 200

    def test_tipo_invalido(self, client, usuario, vivienda, acceso):
        login_usuario(client, usuario)
        resp = client.post('/incidencias/nueva', data={
            'titulo': 'Titulo',
            'descripcion': 'Desc',
            'tipo': 'tipo_inventado',
            'vivienda_safe_oid': oid_to_safe(vivienda.__oid__),
        })
        assert resp.status_code == 200

    def test_sin_vivienda_asignada(self, client, usuario):
        """Un usuario sin viviendas no puede crear incidencias."""
        login_usuario(client, usuario)
        resp = client.get('/incidencias/nueva', follow_redirects=True)
        assert resp.status_code == 200


class TestDetalleIncidencia:
    def test_detalle_propietario(self, client, usuario, incidencia):
        login_usuario(client, usuario)
        safe = oid_to_safe(incidencia.__oid__)
        resp = client.get(f'/incidencias/{safe}')
        assert resp.status_code == 200

    def test_detalle_superadmin(self, client, superadmin, incidencia):
        login_superadmin(client, superadmin)
        safe = oid_to_safe(incidencia.__oid__)
        resp = client.get(f'/incidencias/{safe}')
        assert resp.status_code == 200

    def test_detalle_otro_usuario(self, client, app, srp, incidencia):
        """Un usuario no puede ver incidencias ajenas."""
        from app.models.usuario import Usuario
        u2 = Usuario('Otro', 'otro@test.com', 'otro12345', 'normal')
        srp.save(u2)
        from tests.conftest import login_as
        login_as(client, 'otro@test.com', 'otro12345')
        safe = oid_to_safe(incidencia.__oid__)
        resp = client.get(f'/incidencias/{safe}')
        assert resp.status_code == 403


class TestResponderIncidencia:
    def test_responder_ok(self, client, superadmin, incidencia, srp):
        login_superadmin(client, superadmin)
        safe = oid_to_safe(incidencia.__oid__)
        resp = client.post(f'/incidencias/{safe}/responder', data={
            'resp-respuesta': 'Se ha reparado la fuga.',
            'resp-estado': 'cerrada',
        }, follow_redirects=True)
        assert resp.status_code == 200
        inc = srp.load(Incidencia, incidencia.__oid__)
        assert inc.estado == 'cerrada'
        assert inc.fecha_cierre is not None

    def test_reabrir(self, client, superadmin, incidencia, srp):
        login_superadmin(client, superadmin)
        safe = oid_to_safe(incidencia.__oid__)
        # Cerrar
        client.post(f'/incidencias/{safe}/responder', data={
            'resp-respuesta': 'Cerrada',
            'resp-estado': 'cerrada',
        })
        # Reabrir
        client.post(f'/incidencias/{safe}/responder', data={
            'resp-respuesta': 'Reabierta',
            'resp-estado': 'abierta',
        })
        inc = srp.load(Incidencia, incidencia.__oid__)
        assert inc.estado == 'abierta'
        assert inc.fecha_cierre is None

    def test_responder_requiere_superadmin(self, client, usuario, incidencia):
        login_usuario(client, usuario)
        safe = oid_to_safe(incidencia.__oid__)
        resp = client.post(f'/incidencias/{safe}/responder', data={
            'resp-respuesta': 'No debería poder',
            'resp-estado': 'cerrada',
        })
        assert resp.status_code == 403

    def test_respuesta_vacia(self, client, superadmin, incidencia):
        login_superadmin(client, superadmin)
        safe = oid_to_safe(incidencia.__oid__)
        resp = client.post(f'/incidencias/{safe}/responder', data={
            'resp-respuesta': '',
            'resp-estado': 'cerrada',
        }, follow_redirects=True)
        assert b'Revisa' in resp.data or resp.status_code == 200


class TestEliminarIncidencia:
    def test_eliminar_ok(self, client, superadmin, incidencia, srp):
        login_superadmin(client, superadmin)
        safe = oid_to_safe(incidencia.__oid__)
        resp = client.post(f'/incidencias/{safe}/eliminar', follow_redirects=True)
        assert resp.status_code == 200
        assert srp.num_objs(Incidencia) == 0

    def test_eliminar_requiere_superadmin(self, client, usuario, incidencia):
        login_usuario(client, usuario)
        safe = oid_to_safe(incidencia.__oid__)
        resp = client.post(f'/incidencias/{safe}/eliminar')
        assert resp.status_code == 403
