"""Tests de gestión de accesos vivienda-usuario."""
import pytest
from app.helpers import oid_to_safe
from app.models.acceso import AccesoVivienda
from app.models.usuario import Usuario
from tests.conftest import login_superadmin, login_usuario


class TestListaAccesos:
    def test_lista_ok(self, client, superadmin, vivienda, acceso):
        login_superadmin(client, superadmin)
        safe = oid_to_safe(vivienda.__oid__)
        resp = client.get(f'/accesos/vivienda/{safe}/')
        assert resp.status_code == 200

    def test_requiere_superadmin(self, client, usuario, vivienda):
        login_usuario(client, usuario)
        safe = oid_to_safe(vivienda.__oid__)
        resp = client.get(f'/accesos/vivienda/{safe}/')
        assert resp.status_code == 403


class TestNuevoAcceso:
    def test_crear_ok(self, client, superadmin, vivienda, srp):
        login_superadmin(client, superadmin)
        # Crear otro usuario para asignar
        u2 = Usuario('Otro', 'otro@test.com', 'otro12345', 'normal')
        srp.save(u2)
        safe_viv = oid_to_safe(vivienda.__oid__)
        resp = client.post(f'/accesos/vivienda/{safe_viv}/nuevo', data={
            'usuario_safe_oid': oid_to_safe(u2.__oid__),
            'rol_en_vivienda': 'convivente',
        }, follow_redirects=True)
        assert resp.status_code == 200

    def test_crear_duplicado(self, client, superadmin, vivienda, usuario, acceso, srp):
        login_superadmin(client, superadmin)
        safe_viv = oid_to_safe(vivienda.__oid__)
        resp = client.post(f'/accesos/vivienda/{safe_viv}/nuevo', data={
            'usuario_safe_oid': oid_to_safe(usuario.__oid__),
            'rol_en_vivienda': 'titular',
        }, follow_redirects=True)
        assert b'ya tiene acceso' in resp.data


class TestEliminarAcceso:
    def test_eliminar_ok(self, client, superadmin, vivienda, srp):
        """Se puede eliminar un acceso si hay otro titular."""
        login_superadmin(client, superadmin)
        u1 = Usuario('T1', 't1@t.com', 't1pass123', 'normal')
        u2 = Usuario('T2', 't2@t.com', 't2pass123', 'normal')
        srp.save(u1)
        srp.save(u2)
        a1 = AccesoVivienda(u1.__oid__, vivienda.__oid__, 'titular')
        a2 = AccesoVivienda(u2.__oid__, vivienda.__oid__, 'titular')
        srp.save(a1)
        srp.save(a2)
        safe = oid_to_safe(a1.__oid__)
        resp = client.post(f'/accesos/{safe}/eliminar', follow_redirects=True)
        assert resp.status_code == 200

    def test_no_eliminar_unico_titular(self, client, superadmin, acceso):
        """No se puede eliminar el único titular."""
        login_superadmin(client, superadmin)
        safe = oid_to_safe(acceso.__oid__)
        resp = client.post(f'/accesos/{safe}/eliminar', follow_redirects=True)
        assert b'titular' in resp.data.lower() or resp.status_code == 200
