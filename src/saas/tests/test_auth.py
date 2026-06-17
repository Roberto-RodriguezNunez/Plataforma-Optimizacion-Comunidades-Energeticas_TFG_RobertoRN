"""Tests de autenticación: login, registro, perfil, cambiar contraseña."""
import pytest
from tests.conftest import login_as, login_superadmin, login_usuario
from app.models.usuario import Usuario


# ══════════════════════════════════════════════════════════════════
# LOGIN
# ══════════════════════════════════════════════════════════════════

class TestLogin:
    def test_login_get(self, client):
        resp = client.get('/login')
        assert resp.status_code == 200
        assert b'Iniciar' in resp.data

    def test_login_ok(self, client, superadmin):
        resp = login_as(client, 'admin@test.com', 'admin1234')
        assert resp.status_code == 200

    def test_login_wrong_password(self, client, superadmin):
        resp = login_as(client, 'admin@test.com', 'wrongpass')
        assert b'incorrectos' in resp.data

    def test_login_wrong_email(self, client, superadmin):
        resp = login_as(client, 'nope@test.com', 'admin1234')
        assert b'incorrectos' in resp.data

    def test_login_empty_email(self, client):
        resp = client.post('/login', data={'email': '', 'password': 'x'})
        assert resp.status_code == 200  # form re-rendered with errors

    def test_login_empty_password(self, client, superadmin):
        resp = client.post('/login', data={'email': 'admin@test.com', 'password': ''})
        assert resp.status_code == 200

    def test_login_invalid_email_format(self, client):
        resp = client.post('/login', data={'email': 'not-an-email', 'password': 'x'})
        assert resp.status_code == 200

    def test_login_redirect_if_authenticated(self, client, superadmin):
        login_superadmin(client, superadmin)
        resp = client.get('/login')
        assert resp.status_code == 302

    def test_logout(self, client, superadmin):
        login_superadmin(client, superadmin)
        resp = client.get('/logout', follow_redirects=True)
        assert resp.status_code == 200


# ══════════════════════════════════════════════════════════════════
# REGISTRO
# ══════════════════════════════════════════════════════════════════

class TestRegistro:
    def test_registro_get(self, client):
        resp = client.get('/registro')
        assert resp.status_code == 200

    def test_registro_ok(self, client, app, srp):
        resp = client.post('/registro', data={
            'nombre': 'Nuevo User',
            'email': 'nuevo@test.com',
            'password': 'nuevo1234',
            'password2': 'nuevo1234',
        }, follow_redirects=True)
        assert resp.status_code == 200
        u = srp.find_first(Usuario, lambda u: u.email == 'nuevo@test.com')
        assert u is not None

    def test_primer_usuario_es_superadmin(self, client, app, srp):
        """El primer usuario registrado debe ser superadmin."""
        client.post('/registro', data={
            'nombre': 'Primero',
            'email': 'primero@test.com',
            'password': 'primero1234',
            'password2': 'primero1234',
        })
        u = srp.find_first(Usuario, lambda u: u.email == 'primero@test.com')
        assert u.rol_global == 'superadmin'

    def test_segundo_usuario_es_normal(self, client, app, srp, superadmin):
        """Los siguientes usuarios son normales."""
        client.post('/registro', data={
            'nombre': 'Segundo',
            'email': 'segundo@test.com',
            'password': 'segundo1234',
            'password2': 'segundo1234',
        })
        u = srp.find_first(Usuario, lambda u: u.email == 'segundo@test.com')
        assert u.rol_global == 'normal'

    def test_registro_email_duplicado(self, client, superadmin):
        resp = client.post('/registro', data={
            'nombre': 'Dup',
            'email': 'admin@test.com',
            'password': 'dup12345678',
            'password2': 'dup12345678',
        }, follow_redirects=True)
        assert b'ya est' in resp.data

    def test_registro_password_corta(self, client):
        resp = client.post('/registro', data={
            'nombre': 'Short',
            'email': 'short@test.com',
            'password': '123',
            'password2': '123',
        })
        assert resp.status_code == 200  # form re-rendered

    def test_registro_passwords_no_coinciden(self, client):
        resp = client.post('/registro', data={
            'nombre': 'Mismatch',
            'email': 'mis@test.com',
            'password': 'password1234',
            'password2': 'differentpass',
        })
        assert resp.status_code == 200

    def test_registro_nombre_vacio(self, client):
        resp = client.post('/registro', data={
            'nombre': '',
            'email': 'empty@test.com',
            'password': 'pass12345',
            'password2': 'pass12345',
        })
        assert resp.status_code == 200

    def test_registro_nombre_muy_corto(self, client):
        resp = client.post('/registro', data={
            'nombre': 'A',
            'email': 'a@test.com',
            'password': 'pass12345',
            'password2': 'pass12345',
        })
        assert resp.status_code == 200

    def test_registro_redirect_si_autenticado(self, client, superadmin):
        login_superadmin(client, superadmin)
        resp = client.get('/registro')
        assert resp.status_code == 302


# ══════════════════════════════════════════════════════════════════
# PERFIL
# ══════════════════════════════════════════════════════════════════

class TestPerfil:
    def test_perfil_requiere_login(self, client):
        resp = client.get('/perfil')
        assert resp.status_code == 302

    def test_perfil_get(self, client, superadmin):
        login_superadmin(client, superadmin)
        resp = client.get('/perfil')
        assert resp.status_code == 200

    def test_perfil_actualizar_nombre(self, client, superadmin, srp):
        login_superadmin(client, superadmin)
        resp = client.post('/perfil', data={'nombre': 'Nuevo Nombre'},
                           follow_redirects=True)
        assert resp.status_code == 200


# ══════════════════════════════════════════════════════════════════
# CAMBIAR CONTRASEÑA
# ══════════════════════════════════════════════════════════════════

class TestCambiarPassword:
    def test_cambiar_password_requiere_login(self, client):
        resp = client.get('/cambiar-password')
        assert resp.status_code == 302

    def test_cambiar_password_ok(self, client, superadmin):
        login_superadmin(client, superadmin)
        resp = client.post('/cambiar-password', data={
            'password_actual': 'admin1234',
            'nueva_password': 'newpass1234',
            'nueva_password2': 'newpass1234',
        }, follow_redirects=True)
        assert resp.status_code == 200

    def test_cambiar_password_actual_incorrecta(self, client, superadmin):
        login_superadmin(client, superadmin)
        resp = client.post('/cambiar-password', data={
            'password_actual': 'wrong',
            'nueva_password': 'newpass1234',
            'nueva_password2': 'newpass1234',
        }, follow_redirects=True)
        assert b'actual no es correcta' in resp.data

    def test_cambiar_password_no_coinciden(self, client, superadmin):
        login_superadmin(client, superadmin)
        resp = client.post('/cambiar-password', data={
            'password_actual': 'admin1234',
            'nueva_password': 'newpass1234',
            'nueva_password2': 'different123',
        })
        assert resp.status_code == 200

    def test_cambiar_password_nueva_corta(self, client, superadmin):
        login_superadmin(client, superadmin)
        resp = client.post('/cambiar-password', data={
            'password_actual': 'admin1234',
            'nueva_password': '123',
            'nueva_password2': '123',
        })
        assert resp.status_code == 200
