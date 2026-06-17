"""Tests de notificaciones."""
import pytest
from app.helpers import oid_to_safe
from app.models.notificacion import Notificacion
from app.models.usuario import Usuario
from tests.conftest import login_superadmin, login_usuario, login_as


class TestListaNotificaciones:
    def test_lista_ok(self, client, usuario, notificacion):
        login_usuario(client, usuario)
        resp = client.get('/notificaciones/')
        assert resp.status_code == 200
        assert b'Notif Test' in resp.data

    def test_requiere_login(self, client):
        resp = client.get('/notificaciones/')
        assert resp.status_code == 302


class TestMarcarLeida:
    def test_marcar_ok(self, client, usuario, notificacion, srp):
        login_usuario(client, usuario)
        safe = oid_to_safe(notificacion.__oid__)
        resp = client.post(f'/notificaciones/{safe}/leer', follow_redirects=True)
        assert resp.status_code == 200
        n = srp.load(Notificacion, notificacion.__oid__)
        assert n.leida is True

    def test_marcar_ajax(self, client, usuario, notificacion):
        login_usuario(client, usuario)
        safe = oid_to_safe(notificacion.__oid__)
        resp = client.post(f'/notificaciones/{safe}/leer',
                           headers={'X-Requested-With': 'XMLHttpRequest'})
        assert resp.json['success'] is True

    def test_marcar_ajena(self, client, app, srp, notificacion):
        """Un usuario no puede marcar como leída la notificación de otro."""
        u2 = Usuario('Otro', 'otro@test.com', 'otro12345', 'normal')
        srp.save(u2)
        login_as(client, 'otro@test.com', 'otro12345')
        safe = oid_to_safe(notificacion.__oid__)
        resp = client.post(f'/notificaciones/{safe}/leer')
        assert resp.status_code == 403


class TestMarcarTodasLeidas:
    def test_marcar_todas(self, client, usuario, srp):
        login_usuario(client, usuario)
        # Crear varias notificaciones
        for i in range(3):
            n = Notificacion(usuario.__oid__, 'general', f'N{i}', f'Msg{i}')
            srp.save(n)
        resp = client.post('/notificaciones/leer-todas', follow_redirects=True)
        assert resp.status_code == 200

    def test_marcar_todas_ajax(self, client, usuario, notificacion):
        login_usuario(client, usuario)
        resp = client.post('/notificaciones/leer-todas',
                           headers={'X-Requested-With': 'XMLHttpRequest'})
        assert resp.json['success'] is True


class TestEliminarNotificacion:
    def test_eliminar_propia(self, client, usuario, notificacion, srp):
        login_usuario(client, usuario)
        safe = oid_to_safe(notificacion.__oid__)
        resp = client.post(f'/notificaciones/{safe}/eliminar', follow_redirects=True)
        assert resp.status_code == 200
        assert srp.num_objs(Notificacion) == 0

    def test_eliminar_ajena(self, client, app, srp, notificacion):
        u2 = Usuario('Otro', 'otro@test.com', 'otro12345', 'normal')
        srp.save(u2)
        login_as(client, 'otro@test.com', 'otro12345')
        safe = oid_to_safe(notificacion.__oid__)
        resp = client.post(f'/notificaciones/{safe}/eliminar')
        assert resp.status_code == 403

    def test_superadmin_puede_eliminar_ajena(self, client, superadmin, notificacion, srp):
        login_superadmin(client, superadmin)
        safe = oid_to_safe(notificacion.__oid__)
        resp = client.post(f'/notificaciones/{safe}/eliminar', follow_redirects=True)
        assert resp.status_code == 200


class TestNuevaNotificacion:
    def test_get_formulario(self, client, superadmin):
        login_superadmin(client, superadmin)
        resp = client.get('/notificaciones/nueva')
        assert resp.status_code == 200
        assert b'Enviar' in resp.data

    def test_enviar_a_usuario(self, client, superadmin, usuario, srp):
        login_superadmin(client, superadmin)
        safe = oid_to_safe(usuario.__oid__)
        resp = client.post('/notificaciones/nueva', data={
            'destinatario': safe,
            'titulo': 'Aviso importante',
            'mensaje': 'Contenido del aviso',
        }, follow_redirects=True)
        assert resp.status_code == 200
        n = srp.find_first(Notificacion, lambda n: n.titulo == 'Aviso importante')
        assert n is not None
        assert str(n.usuario_oid) == str(usuario.__oid__)

    def test_enviar_a_todos(self, client, superadmin, usuario, srp):
        login_superadmin(client, superadmin)
        resp = client.post('/notificaciones/nueva', data={
            'destinatario': '__todos__',
            'titulo': 'Aviso general',
            'mensaje': 'Para todos los usuarios',
        }, follow_redirects=True)
        assert resp.status_code == 200
        notifs = [n for n in srp.load_all(Notificacion) if n.titulo == 'Aviso general']
        assert len(notifs) == 2  # superadmin + usuario

    def test_requiere_superadmin(self, client, usuario):
        login_usuario(client, usuario)
        resp = client.get('/notificaciones/nueva')
        assert resp.status_code == 403

    def test_titulo_vacio(self, client, superadmin, usuario):
        login_superadmin(client, superadmin)
        resp = client.post('/notificaciones/nueva', data={
            'destinatario': '__todos__',
            'titulo': '',
            'mensaje': 'Contenido',
        })
        assert resp.status_code == 200  # form re-rendered

    def test_mensaje_vacio(self, client, superadmin, usuario):
        login_superadmin(client, superadmin)
        resp = client.post('/notificaciones/nueva', data={
            'destinatario': '__todos__',
            'titulo': 'Título',
            'mensaje': '',
        })
        assert resp.status_code == 200

    def test_titulo_muy_corto(self, client, superadmin):
        login_superadmin(client, superadmin)
        resp = client.post('/notificaciones/nueva', data={
            'destinatario': '__todos__',
            'titulo': 'A',
            'mensaje': 'Contenido válido',
        })
        assert resp.status_code == 200
