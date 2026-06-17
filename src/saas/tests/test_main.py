"""Tests de la ruta principal / dashboard."""
import pytest
from tests.conftest import login_superadmin, login_usuario


class TestIndex:
    def test_landing_sin_sesion(self, client):
        resp = client.get('/')
        assert resp.status_code == 200
        assert b'LeaLink' in resp.data

    def test_dashboard_superadmin(self, client, superadmin):
        login_superadmin(client, superadmin)
        resp = client.get('/')
        assert resp.status_code == 200

    def test_dashboard_usuario(self, client, usuario, vivienda, acceso, comunidad):
        login_usuario(client, usuario)
        resp = client.get('/')
        assert resp.status_code == 200

    def test_dashboard_usuario_sin_viviendas(self, client, usuario):
        login_usuario(client, usuario)
        resp = client.get('/')
        assert resp.status_code == 200


class TestErrorPages:
    def test_404(self, client, superadmin):
        login_superadmin(client, superadmin)
        resp = client.get('/ruta-que-no-existe')
        assert resp.status_code == 404

    def test_403_usuario_en_superadmin_route(self, client, usuario):
        login_usuario(client, usuario)
        resp = client.get('/comunidades/')
        assert resp.status_code == 403
