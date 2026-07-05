"""Tests de generación de CierreMensual desde OperacionHoraria."""
import math
import random as _rng
from datetime import datetime, timezone

import pytest
from app.extensions import db
from app.helpers import oid_to_safe
from app.models.cierre import CierreMensual
from app.models.operacion import OperacionHoraria
from app.models.vivienda import Vivienda
from app.modules.cierres.routes import _distribuir
from tests.conftest import login_superadmin, login_usuario


# ---------------------------------------------------------------------------
# Tests unitarios de _distribuir (sin Flask, sin DB)
# ---------------------------------------------------------------------------

class TestDistribuir:
    def test_suma_igual_total(self):
        total = 42.7
        pesos = [0.3, 0.5, 0.2]
        ids   = [1, 2, 3]
        result = _distribuir(total, pesos, step=7, vivienda_ids=ids, sigma=0.15)
        assert abs(sum(result) - total) < 1e-9

    def test_todos_positivos(self):
        result = _distribuir(10.0, [0.1, 0.4, 0.5], step=0, vivienda_ids=[10, 20, 30], sigma=0.15)
        assert all(x > 0 for x in result)

    def test_reproducible(self):
        """Mismo seed → mismo resultado."""
        args = ([0.4, 0.3, 0.3], 5, [1, 2, 3], 0.15)
        r1 = _distribuir(20.0, *args)
        r2 = _distribuir(20.0, *args)
        assert r1 == r2

    def test_una_sola_vivienda(self):
        """Con una sola vivienda, sin importar el ruido, devuelve el total."""
        result = _distribuir(15.3, [1.0], step=99, vivienda_ids=[7], sigma=0.15)
        assert abs(result[0] - 15.3) < 1e-9

    def test_total_cero(self):
        result = _distribuir(0.0, [0.5, 0.5], step=1, vivienda_ids=[1, 2], sigma=0.15)
        assert all(x == 0.0 for x in result)

    def test_mayor_peso_mayor_media(self):
        """La vivienda con mayor peso base recibe más en media (muchos steps)."""
        pesos = [0.8, 0.2]
        ids   = [1, 2]
        sumas = [0.0, 0.0]
        for step in range(500):
            r = _distribuir(10.0, pesos, step, ids, sigma=0.15)
            sumas[0] += r[0]
            sumas[1] += r[1]
        assert sumas[0] > sumas[1]

    def test_mayor_kwp_mayor_media(self):
        """Vivienda con más kwp recibe más generación en media."""
        total_kwp = 6.4 + 3.6
        gen_pesos = [6.4 / total_kwp, 3.6 / total_kwp]
        sumas = [0.0, 0.0]
        for step in range(300):
            r = _distribuir(10.0, gen_pesos, step, [1, 2], sigma=0.10)
            sumas[0] += r[0]
            sumas[1] += r[1]
        assert sumas[0] > sumas[1], "mayor kwp debe recibir más generación en media"


# ---------------------------------------------------------------------------
# Helpers para tests de integración
# ---------------------------------------------------------------------------

def _insertar_operaciones(comunidad_oid, n=48, mes='2025-01'):
    year, month = int(mes[:4]), int(mes[5:7])
    for i in range(n):
        ts = datetime(year, month, 1 + i // 24, i % 24, tzinfo=timezone.utc)
        op = OperacionHoraria(
            comunidad_oid=comunidad_oid,
            ts=ts, step=i,
            consumo_total_kwh=10.0, gen_total_kwh=4.0,
            precio_compra=0.20, precio_exc=0.06,
            soc=0.6,
            p_carga_solar=2.0, p_carga_red=0.0,
            p_descarga_casa=1.5, p_descarga_red=0.0,
            beneficio_marginal=0.15,
        )
        db.session.add(op)
    db.session.commit()


def _vivienda_con_paneles(comunidad_oid, coef=1.0, kwp=4.0, cups='ES0001', orient='sur'):
    v = Vivienda(
        comunidad_oid=comunidad_oid,
        identificador=f'V-{cups}',
        cups=cups,
        potencia_contratada_kw=4.6,
        coeficiente_reparto=coef,
        tiene_paneles=True,
        potencia_pico_paneles_kwp=kwp,
        numero_paneles=10,
        orientacion_paneles=orient,
    )
    db.session.add(v)
    db.session.commit()
    return v


# ---------------------------------------------------------------------------
# Tests de integración con Flask/DB
# ---------------------------------------------------------------------------

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
        _vivienda_con_paneles(comunidad.__oid__)
        _insertar_operaciones(comunidad.__oid__, n=48, mes='2025-01')

        login_superadmin(client, superadmin)
        safe = oid_to_safe(comunidad.__oid__)
        resp = client.post(f'/cierres/comunidad/{safe}/generar/2025-01',
                           follow_redirects=True)
        assert resp.status_code == 200
        assert srp.num_objs(CierreMensual) == 1

        c = CierreMensual.query.first()
        assert c.mes == '2025-01'
        assert c.consumo_total_kwh > 0
        assert c.autoconsumo_directo_kwh >= 0
        assert c.ahorro_eur >= 0

    def test_formula_ahorro_coherente_una_vivienda(self, app, client, superadmin, comunidad):
        """Una sola vivienda: _distribuir devuelve el total exacto → valores exactos.

        consumo=10, gen=4, p_descarga_casa=1.5, pc=0.20, pe=0.06:
          factura_base (sin comunidad) = max(0, 10−4)×0.20 = 1.20
          factura neteada con batería  = max(0, 10−4−1.5)×0.20 = 0.90
          ahorro_total = 1.20 − 0.90 = 0.30 → coef=1.0 → ahorro=0.30, cuota=0.90
        """
        _vivienda_con_paneles(comunidad.__oid__, coef=1.0, kwp=4.0)
        op = OperacionHoraria(
            comunidad_oid=comunidad.__oid__, ts=datetime(2025, 1, 1, 0, tzinfo=timezone.utc),
            step=0, consumo_total_kwh=10.0, gen_total_kwh=4.0,
            precio_compra=0.20, precio_exc=0.06, soc=0.6,
            p_carga_solar=0.0, p_carga_red=0.0, p_descarga_casa=1.5,
            p_descarga_red=0.0, beneficio_marginal=0.15,
        )
        db.session.add(op)
        db.session.commit()

        login_superadmin(client, superadmin)
        safe = oid_to_safe(comunidad.__oid__)
        client.post(f'/cierres/comunidad/{safe}/generar/2025-01', follow_redirects=True)

        c = CierreMensual.query.first()
        assert c is not None
        assert abs(c.consumo_total_kwh - 10.0) < 1e-4
        assert abs(c.autoconsumo_directo_kwh - 4.0) < 1e-4
        assert abs(c.energia_de_bateria_kwh - 1.5) < 1e-4
        assert abs(c.factura_sin_paneles_eur - 2.00) < 1e-3
        assert abs(c.factura_escenario_base_eur - 1.20) < 1e-3
        assert abs(c.factura_escenario_real_eur - 0.90) < 1e-3
        assert abs(c.ahorro_eur - 0.30) < 1e-3

    def test_suma_cuotas_igual_factura_neteada(self, app, client, superadmin, comunidad):
        """La suma de cuotas por vivienda reproduce la factura común neteada.

        Con las operaciones de _insertar_operaciones (ct=10, gt=4, pd_casa=1.5,
        p_carga_solar=2, pc=0.20): compra neteada = 4.5 kWh/h → 0.90 €/h.
        """
        _vivienda_con_paneles(comunidad.__oid__, coef=0.5, kwp=4.0, cups='ES0030')
        _vivienda_con_paneles(comunidad.__oid__, coef=0.3, kwp=2.4, cups='ES0031')
        _vivienda_con_paneles(comunidad.__oid__, coef=0.2, kwp=1.6, cups='ES0032')
        _insertar_operaciones(comunidad.__oid__, n=24, mes='2025-01')

        login_superadmin(client, superadmin)
        safe = oid_to_safe(comunidad.__oid__)
        client.post(f'/cierres/comunidad/{safe}/generar/2025-01', follow_redirects=True)

        cierres = CierreMensual.query.all()
        assert len(cierres) == 3
        factura_neteada = 24 * 0.90
        suma_cuotas = sum(c.factura_escenario_real_eur for c in cierres)
        assert abs(suma_cuotas - factura_neteada) < 0.05, \
            f"Σ cuotas {suma_cuotas:.2f} ≠ factura neteada {factura_neteada:.2f}"
        # El ahorro de cada vivienda es proporcional a su coeficiente
        ahorro_total = sum(c.ahorro_eur for c in cierres)
        for c in cierres:
            assert abs(c.ahorro_eur - c.coeficiente_reparto_aplicado * ahorro_total) < 0.05

    def test_suma_consumos_individuales_igual_total(self, app, client, superadmin, comunidad, srp):
        """La suma de consumos individuales debe ser ≈ consumo_total acumulado."""
        v1 = _vivienda_con_paneles(comunidad.__oid__, coef=0.4, kwp=3.2, cups='ES0001', orient='sur')
        v2 = _vivienda_con_paneles(comunidad.__oid__, coef=0.35, kwp=2.8, cups='ES0002', orient='mixta')
        v3 = _vivienda_con_paneles(comunidad.__oid__, coef=0.25, kwp=2.0, cups='ES0003', orient='este')
        _insertar_operaciones(comunidad.__oid__, n=24, mes='2025-01')

        login_superadmin(client, superadmin)
        safe = oid_to_safe(comunidad.__oid__)
        client.post(f'/cierres/comunidad/{safe}/generar/2025-01', follow_redirects=True)

        cierres = CierreMensual.query.all()
        assert len(cierres) == 3

        suma_consumo = sum(c.consumo_total_kwh for c in cierres)
        consumo_esperado = 10.0 * 24   # 24 horas × 10 kWh/h
        # Tolerancia: round(..., 3) introduce hasta 0.0005 por vivienda × 3 = 0.0015
        assert abs(suma_consumo - consumo_esperado) < 0.01, \
            f"Suma consumos individuales {suma_consumo:.4f} ≠ total {consumo_esperado}"

    def test_suma_generaciones_individuales_igual_total(self, app, client, superadmin, comunidad, srp):
        """autoconsumo + vertido por casa debe sumar a gen_total acumulado (sin batería)."""
        _vivienda_con_paneles(comunidad.__oid__, coef=0.6, kwp=4.0, cups='ES0010', orient='sur')
        _vivienda_con_paneles(comunidad.__oid__, coef=0.4, kwp=2.0, cups='ES0011', orient='este')
        # Operación sin batería (p_descarga_casa=0) para que gen = autoconsumo + vertido exacto
        op = OperacionHoraria(
            comunidad_oid=comunidad.__oid__, ts=datetime(2025, 1, 1, 0, tzinfo=timezone.utc),
            step=0, consumo_total_kwh=5.0, gen_total_kwh=8.0,
            precio_compra=0.20, precio_exc=0.06, soc=0.5,
            p_carga_solar=0.0, p_carga_red=0.0, p_descarga_casa=0.0,
            p_descarga_red=0.0, beneficio_marginal=0.10,
        )
        db.session.add(op)
        db.session.commit()

        login_superadmin(client, superadmin)
        safe = oid_to_safe(comunidad.__oid__)
        client.post(f'/cierres/comunidad/{safe}/generar/2025-01', follow_redirects=True)

        cierres = CierreMensual.query.all()
        suma_gen = sum(c.autoconsumo_directo_kwh + c.vertido_a_red_kwh for c in cierres)
        assert abs(suma_gen - 8.0) < 1e-3, \
            f"Suma generaciones individuales {suma_gen:.3f} ≠ gen_total 8.0"

    def test_mayor_kwp_recibe_mas_generacion(self, app, client, superadmin, comunidad, srp):
        """Vivienda con más kwp instalado recibe más generación atribuida en media."""
        v_grande = _vivienda_con_paneles(comunidad.__oid__, coef=0.5, kwp=6.4, cups='ES0020', orient='sur')
        v_pequena = _vivienda_con_paneles(comunidad.__oid__, coef=0.5, kwp=3.2, cups='ES0021', orient='sur')
        _insertar_operaciones(comunidad.__oid__, n=48, mes='2025-01')

        login_superadmin(client, superadmin)
        safe = oid_to_safe(comunidad.__oid__)
        client.post(f'/cierres/comunidad/{safe}/generar/2025-01', follow_redirects=True)

        c_g = CierreMensual.query.filter_by(vivienda_oid=v_grande.id).first()
        c_p = CierreMensual.query.filter_by(vivienda_oid=v_pequena.id).first()
        gen_g = c_g.autoconsumo_directo_kwh + c_g.vertido_a_red_kwh
        gen_p = c_p.autoconsumo_directo_kwh + c_p.vertido_a_red_kwh
        assert gen_g > gen_p, "Más kwp instalado → más generación atribuida"

    def test_cierre_existente_se_actualiza(self, app, client, superadmin, comunidad, srp):
        """Segunda llamada actualiza el cierre en lugar de crear uno nuevo."""
        _vivienda_con_paneles(comunidad.__oid__)
        _insertar_operaciones(comunidad.__oid__, n=24, mes='2025-01')
        login_superadmin(client, superadmin)
        safe = oid_to_safe(comunidad.__oid__)

        client.post(f'/cierres/comunidad/{safe}/generar/2025-01', follow_redirects=True)
        assert srp.num_objs(CierreMensual) == 1

        client.post(f'/cierres/comunidad/{safe}/generar/2025-01', follow_redirects=True)
        assert srp.num_objs(CierreMensual) == 1   # no duplica
