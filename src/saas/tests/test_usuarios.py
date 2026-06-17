"""Tests de gestión de usuarios."""
import pytest
from app.helpers import oid_to_safe
from app.models.acceso import AccesoVivienda
from tests.conftest import login_superadmin, login_usuario


class TestListaUsuarios:
    def test_requiere_superadmin(self, client, usuario):
        login_usuario(client, usuario)
        resp = client.get('/usuarios/')
        assert resp.status_code == 403

    def test_lista_ok(self, client, superadmin):
        login_superadmin(client, superadmin)
        resp = client.get('/usuarios/')
        assert resp.status_code == 200

    def test_busqueda(self, client, superadmin, usuario):
        login_superadmin(client, superadmin)
        resp = client.get('/usuarios/?q=user')
        assert resp.status_code == 200


class TestDetalleUsuario:
    def test_detalle_ok(self, client, superadmin, usuario):
        login_superadmin(client, superadmin)
        safe = oid_to_safe(usuario.__oid__)
        resp = client.get(f'/usuarios/{safe}')
        assert resp.status_code == 200
        assert b'User Test' in resp.data

    def test_detalle_requiere_superadmin(self, client, usuario):
        login_usuario(client, usuario)
        safe = oid_to_safe(usuario.__oid__)
        resp = client.get(f'/usuarios/{safe}')
        assert resp.status_code == 403

    def test_detalle_oid_invalido(self, client, superadmin):
        login_superadmin(client, superadmin)
        resp = client.get('/usuarios/oid-falso')
        assert resp.status_code == 404


class TestAsignarVivienda:
    def test_asignar_ok(self, client, superadmin, usuario, vivienda, srp):
        login_superadmin(client, superadmin)
        safe_usr = oid_to_safe(usuario.__oid__)
        safe_viv = oid_to_safe(vivienda.__oid__)
        resp = client.post(f'/usuarios/{safe_usr}/asignar-vivienda', data={
            'vivienda_safe_oid': safe_viv,
            'rol_en_vivienda': 'titular',
        }, follow_redirects=True)
        assert resp.status_code == 200
        assert srp.num_objs(AccesoVivienda) == 1

    def test_asignar_duplicado(self, client, superadmin, usuario, vivienda, acceso):
        login_superadmin(client, superadmin)
        safe_usr = oid_to_safe(usuario.__oid__)
        safe_viv = oid_to_safe(vivienda.__oid__)
        resp = client.post(f'/usuarios/{safe_usr}/asignar-vivienda', data={
            'vivienda_safe_oid': safe_viv,
            'rol_en_vivienda': 'titular',
        }, follow_redirects=True)
        assert b'ya tiene acceso' in resp.data

    def test_asignar_sin_vivienda(self, client, superadmin, usuario):
        login_superadmin(client, superadmin)
        safe = oid_to_safe(usuario.__oid__)
        resp = client.post(f'/usuarios/{safe}/asignar-vivienda', data={
            'vivienda_safe_oid': '',
            'rol_en_vivienda': 'titular',
        }, follow_redirects=True)
        assert b'Selecciona' in resp.data

    def test_asignar_requiere_superadmin(self, client, usuario, vivienda):
        login_usuario(client, usuario)
        safe = oid_to_safe(usuario.__oid__)
        resp = client.post(f'/usuarios/{safe}/asignar-vivienda', data={
            'vivienda_safe_oid': oid_to_safe(vivienda.__oid__),
            'rol_en_vivienda': 'titular',
        })
        assert resp.status_code == 403


class TestEliminarUsuario:
    def test_eliminar_ok(self, client, superadmin, usuario, srp):
        login_superadmin(client, superadmin)
        safe = oid_to_safe(usuario.__oid__)
        resp = client.post(f'/usuarios/{safe}/eliminar', follow_redirects=True)
        assert resp.status_code == 200

    def test_no_puede_eliminar_propio(self, client, superadmin):
        login_superadmin(client, superadmin)
        safe = oid_to_safe(superadmin.__oid__)
        resp = client.post(f'/usuarios/{safe}/eliminar', follow_redirects=True)
        assert b'tu propia cuenta' in resp.data

    def test_no_puede_eliminar_unico_titular(self, client, superadmin, usuario, vivienda, acceso):
        """No se puede borrar un usuario si es el único titular de una vivienda."""
        login_superadmin(client, superadmin)
        safe = oid_to_safe(usuario.__oid__)
        resp = client.post(f'/usuarios/{safe}/eliminar', follow_redirects=True)
        assert b'titular' in resp.data.lower()

    def test_eliminar_requiere_superadmin(self, client, usuario):
        login_usuario(client, usuario)
        safe = oid_to_safe(usuario.__oid__)
        resp = client.post(f'/usuarios/{safe}/eliminar')
        assert resp.status_code == 403

    def test_eliminar_ajax(self, client, superadmin, usuario):
        login_superadmin(client, superadmin)
        safe = oid_to_safe(usuario.__oid__)
        resp = client.post(f'/usuarios/{safe}/eliminar',
                           headers={'X-Requested-With': 'XMLHttpRequest'})
        assert resp.json['success'] is True
