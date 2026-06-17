"""Tests de generación de CierreMensual desde OperacionHoraria."""
import os
from datetime import datetime, timezone

import pytest
from app.extensions import db
from app.helpers import oid_to_safe
from app.models.cierre import CierreMensual
from app.models.operacion import OperacionHoraria
from app.models.vivienda import Vivienda
from tests.conftest import login_superadmin, login_usuario


def _insertar_operaciones(comunidad_oid, n=48, mes='2025-01'):
    """Inserta n filas de OperacionHoraria para el mes pedido."""
    year, month = int(mes[:4]), int(mes[5:7])
    for i in range(n):
        ts = datetime(year, month, 1 + i // 24, i % 24, tzinfo=timezone.utc)
        op = OperacionHoraria(
            comunidad_oid=comunidad_oid,
            ts=ts,
            step=i,
            consumo_total_kwh=10.0,
            gen_total_kwh=4.0,
            precio_compra=0.20,
            precio_exc=0.06,
            soc=0.6,
            p_carga_solar=2.0,
            p_carga_red=0.0,
            p_descarga_casa=1.5,
            p_descarga_red=0.0,
            beneficio_marginal=0.15,
        )
        db.session.add(op)
    db.session.commit()


def _vivienda_con_paneles(comunidad_oid, coef=1.0, kwp=4.0):
    v = Vivienda(
        comunidad_oid=comunidad_oid,
        identificador='V-Test',
        cups='ES0001',
        potencia_contratada_kw=4.6,
        coeficiente_reparto=coef,
        tiene_paneles=True,
        potencia_pico_paneles_kwp=kwp,
        numero_paneles=10,
        orientacion_paneles='sur',
    )
    db.session.add(v)
    db.session.commit()
    return v


class TestGenerarCierreDesdeOperaciones:
    def test_ruta_requiere_superadmin(self, client, usuario, comunidad):
        login_usuario(client, usuario)
        safe = oid_to_safe(comunidad.__oid__)
        resp = client.post(f'/cierres/comunidad/{safe}/generar/2025-01')
        assert resp.status_code == 403

    def test_sin_operaciones_flash_error(self, client, superadmin, comunidad, vivienda):
        login_superadmin(client, superadmin)
        safe = oid_to_safe(comunidad.__oid__)
        resp = client.post(f'/cierres/comunidad/{safe}/generar/2025-01',
                           follow_redirects=True)
        assert resp.status_code == 200

    def test_genera_cierre_por_vivienda(self, app, client, superadmin, comunidad, srp):
        viv = _vivienda_con_paneles(comunidad.__oid__)
        _insertar_operaciones(comunidad.__oid__, n=48, mes='2025-01')

        login_superadmin(client, superadmin)
        safe = oid_to_safe(comunidad.__oid__)
        resp = client.post(f'/cierres/comunidad/{safe}/generar/2025-01',
                           follow_redirects=True)
        assert resp.status_code == 200
        assert srp.num_objs(CierreMensual) == 1

        c = CierreMensual.query.first()
        assert c.mes == '2025-01'
        assert c.vivienda_oid == viv.id
        assert c.consumo_total_kwh > 0
        assert c.autoconsumo_directo_kwh >= 0
        assert c.ahorro_eur >= 0

    def test_formula_ahorro_coherente(self, app, client, superadmin, comunidad):
        """Con coef=1.0 y kwp/total_kwp=1.0 los valores deben ser exactos."""
        viv = _vivienda_con_paneles(comunidad.__oid__, coef=1.0, kwp=4.0)
        # 1 hora: consumo=10, gen=4, precio_compra=0.20, precio_exc=0.06
        # p_descarga_casa=1.5 → bat_cubierto=1.5×1.0=1.5
        # gen_atrib=4×1.0=4, auto=min(4,10)=4, surplus=0
        # deficit=6, compra_red=max(0,6-1.5)=4.5
        # coste_base=10×0.20=2.0
        # factura_real=4.5×0.20=0.90
        # ahorro=2.0-0.90=1.10 por hora
        n = 1
        ts = datetime(2025, 1, 1, 0, tzinfo=timezone.utc)
        op = OperacionHoraria(
            comunidad_oid=comunidad.__oid__, ts=ts, step=0,
            consumo_total_kwh=10.0, gen_total_kwh=4.0,
            precio_compra=0.20, precio_exc=0.06,
            soc=0.6,
            p_carga_solar=0.0, p_carga_red=0.0,
            p_descarga_casa=1.5, p_descarga_red=0.0,
            beneficio_marginal=0.15,
        )
        db.session.add(op)
        db.session.commit()

        login_superadmin(client, superadmin)
        safe = oid_to_safe(comunidad.__oid__)
        client.post(f'/cierres/comunidad/{safe}/generar/2025-01',
                    follow_redirects=True)

        c = CierreMensual.query.first()
        assert c is not None
        assert abs(c.consumo_total_kwh - 10.0) < 1e-4
        assert abs(c.autoconsumo_directo_kwh - 4.0) < 1e-4
        assert abs(c.energia_de_bateria_kwh - 1.5) < 1e-4
        assert abs(c.ahorro_eur - 1.10) < 1e-3

    def test_cierre_existente_se_actualiza(self, app, client, superadmin, comunidad, srp):
        """Segunda llamada actualiza el cierre en lugar de crear uno nuevo."""
        viv = _vivienda_con_paneles(comunidad.__oid__)
        _insertar_operaciones(comunidad.__oid__, n=24, mes='2025-01')
        login_superadmin(client, superadmin)
        safe = oid_to_safe(comunidad.__oid__)

        client.post(f'/cierres/comunidad/{safe}/generar/2025-01', follow_redirects=True)
        assert srp.num_objs(CierreMensual) == 1

        client.post(f'/cierres/comunidad/{safe}/generar/2025-01', follow_redirects=True)
        assert srp.num_objs(CierreMensual) == 1  # no duplica
