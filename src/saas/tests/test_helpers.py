"""Tests de funciones helper."""
import pytest
from app.helpers import (oid_to_safe, oid_from_safe, recalcular_coeficientes,
                          puede_borrar_usuario, usuario_tiene_acceso,
                          cascade_delete_vivienda, cascade_delete_comunidad)
from app.models.vivienda import Vivienda
from app.models.acceso import AccesoVivienda
from app.models.usuario import Usuario
from app.models.comunidad import Comunidad


class TestOidConversion:
    def test_roundtrip(self, app, srp, comunidad):
        oid = comunidad.__oid__
        safe = oid_to_safe(oid)
        recovered = oid_from_safe(safe)
        assert str(oid) == str(recovered)

    def test_safe_no_tiene_caracteres_especiales(self, app, srp, comunidad):
        safe = oid_to_safe(comunidad.__oid__)
        assert '.' not in safe
        assert '@' not in safe

    def test_from_safe_invalido(self, app):
        with pytest.raises(Exception):
            oid_from_safe('esto-no-es-un-oid')


class TestRecalcularCoeficientes:
    def test_una_vivienda(self, app, srp, comunidad):
        v = Vivienda(comunidad.__oid__, 'V1', potencia_contratada_kw=5.0, coeficiente_reparto=0.0)
        srp.save(v)
        recalcular_coeficientes(comunidad.__oid__)
        v = list(srp.load_all(Vivienda))[0]
        assert abs(v.coeficiente_reparto - 1.0) < 0.001

    def test_varias_viviendas_iguales(self, app, srp, comunidad):
        for i in range(4):
            srp.save(Vivienda(comunidad.__oid__, f'V{i}',
                              potencia_contratada_kw=3.0, coeficiente_reparto=0.0))
        recalcular_coeficientes(comunidad.__oid__)
        vivs = list(srp.load_all(Vivienda))
        for v in vivs:
            assert abs(v.coeficiente_reparto - 0.25) < 0.001
        assert abs(sum(v.coeficiente_reparto for v in vivs) - 1.0) < 0.001

    def test_viviendas_distintas_potencias(self, app, srp, comunidad):
        srp.save(Vivienda(comunidad.__oid__, 'Grande',
                          potencia_contratada_kw=6.0, coeficiente_reparto=0.0))
        srp.save(Vivienda(comunidad.__oid__, 'Pequeña',
                          potencia_contratada_kw=2.0, coeficiente_reparto=0.0))
        recalcular_coeficientes(comunidad.__oid__)
        vivs = list(srp.load_all(Vivienda))
        total = sum(v.coeficiente_reparto for v in vivs)
        assert abs(total - 1.0) < 0.001
        grande = [v for v in vivs if v.identificador == 'Grande'][0]
        assert abs(grande.coeficiente_reparto - 0.75) < 0.001

    def test_comunidad_vacia(self, app, srp, comunidad):
        """No debe fallar con comunidad sin viviendas."""
        recalcular_coeficientes(comunidad.__oid__)  # no crash


class TestPuedeBorrarUsuario:
    def test_puede_borrar_sin_accesos(self, app, srp):
        u = Usuario('Test', 'test@x.com', 'pass12345', 'normal')
        srp.save(u)
        puede, motivo = puede_borrar_usuario(u.__oid__)
        assert puede is True

    def test_no_puede_borrar_unico_titular(self, app, srp, comunidad):
        u = Usuario('Tit', 'tit@x.com', 'pass12345', 'normal')
        srp.save(u)
        v = Vivienda(comunidad.__oid__, 'V', potencia_contratada_kw=3.0)
        srp.save(v)
        a = AccesoVivienda(u.__oid__, v.__oid__, 'titular')
        srp.save(a)
        puede, motivo = puede_borrar_usuario(u.__oid__)
        assert puede is False
        assert 'titular' in motivo.lower()

    def test_puede_borrar_con_otro_titular(self, app, srp, comunidad):
        u1 = Usuario('T1', 't1@x.com', 'pass12345', 'normal')
        u2 = Usuario('T2', 't2@x.com', 'pass12345', 'normal')
        srp.save(u1)
        srp.save(u2)
        v = Vivienda(comunidad.__oid__, 'V', potencia_contratada_kw=3.0)
        srp.save(v)
        srp.save(AccesoVivienda(u1.__oid__, v.__oid__, 'titular'))
        srp.save(AccesoVivienda(u2.__oid__, v.__oid__, 'titular'))
        puede, _ = puede_borrar_usuario(u1.__oid__)
        assert puede is True


class TestCascadeDelete:
    def test_cascade_vivienda(self, app, srp, comunidad):
        v = Vivienda(comunidad.__oid__, 'V', potencia_contratada_kw=3.0)
        srp.save(v)
        u = Usuario('U', 'u@x.com', 'pass12345', 'normal')
        srp.save(u)
        srp.save(AccesoVivienda(u.__oid__, v.__oid__, 'titular'))
        cascade_delete_vivienda(v.__oid__)
        assert srp.num_objs(Vivienda) == 0
        assert srp.num_objs(AccesoVivienda) == 0

    def test_cascade_comunidad(self, app, srp):
        c = Comunidad('C', 'Loc', '2024-01-01', 'activa')
        srp.save(c)
        v = Vivienda(c.__oid__, 'V', potencia_contratada_kw=3.0)
        srp.save(v)
        u = Usuario('U', 'u@x.com', 'pass12345', 'normal')
        srp.save(u)
        srp.save(AccesoVivienda(u.__oid__, v.__oid__, 'titular'))
        cascade_delete_comunidad(c.__oid__)
        assert srp.num_objs(Comunidad) == 0
        assert srp.num_objs(Vivienda) == 0
        assert srp.num_objs(AccesoVivienda) == 0


class TestUsuarioTieneAcceso:
    def test_tiene_acceso(self, app, srp, usuario, vivienda, acceso):
        assert usuario_tiene_acceso(usuario.__oid__, vivienda.__oid__) is True

    def test_no_tiene_acceso(self, app, srp, usuario, vivienda):
        assert usuario_tiene_acceso(usuario.__oid__, vivienda.__oid__) is False
