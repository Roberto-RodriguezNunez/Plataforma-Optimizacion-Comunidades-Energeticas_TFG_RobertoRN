"""CRUD de CierresMensuales."""
import json
import math
import random as _rng

from flask import render_template, redirect, url_for, request, abort, jsonify
from flask_login import login_required, current_user

from app.extensions import db
from app.modules.cierres import cierres_bp
from app.modules.cierres.forms import CierreForm
from app.models.cierre import CierreMensual
from app.models.vivienda import Vivienda
from app.models.acceso import AccesoVivienda
from app.models.operacion import OperacionHoraria
from app.helpers import (oid_from_safe, oid_to_safe, flash_exito, flash_error,
                          is_xhr, usuario_tiene_acceso, crear_notificacion)
from app.decorators import superadmin_required

# Desviación del ruido log-normal para la distribución individual
_SIGMA_CONSUMO = 0.15   # ±15 %: variación conductual entre vecinos
_SIGMA_GEN     = 0.10   # ±10 %: variación por sombras puntuales / temperatura


def _distribuir(total, pesos_norm, step, vivienda_ids, sigma):
    """Distribuye `total` entre N viviendas con ruido log-normal reproducible.

    Cada casa recibe un peso base (coef o kwp×orient) perturbado con ruido
    log-normal de desviación `sigma`, sembrado deterministicamente por
    (vivienda_id, step) para que recalcular el cierre dé siempre el mismo
    resultado. La normalización garantiza sum(devuelto) == total exactamente.
    """
    p = list(pesos_norm)
    for i, vid in enumerate(vivienda_ids):
        r = _rng.Random(int(vid) * 99991 + int(step))
        # Box-Muller: N(0,1) sin dependencias externas
        u1, u2 = max(r.random(), 1e-12), r.random()
        z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        p[i] = max(p[i] * math.exp(sigma * z), 1e-9)
    s = sum(p)
    return [total * x / s for x in p]


def _comprobar_acceso_o_superadmin(viv_oid):
    if current_user.es_superadmin:
        return
    if not usuario_tiene_acceso(current_user.id, viv_oid):
        abort(403)


def _calcular_cierre_desde_operaciones(comunidad_id, mes):
    """Genera o actualiza CierreMensual para cada vivienda a partir de OperacionHoraria.

    Distribución individual (RD 244/2019 + perfil sintético reproducible):

      consumo_i  — distribuido desde consumo_total con ruido log-normal (σ=0.15)
                   ponderado por coef_reparto; sum(consumo_i) == consumo_total exacto.

      gen_i      — distribuido desde gen_total con ruido log-normal (σ=0.10)
                   ponderado por kwp instalado; sum(gen_i) == gen_total exacto.

      bat_i      = p_descarga_casa × coef_reparto_i
      compra_i   = max(0, (consumo_i - gen_i) - bat_i)
      surplus_i  = max(0, gen_i - consumo_i)
      ahorro_i   = consumo_i×pc − (compra_i×pc − surplus_i×pe)

    El ruido está sembrado por (vivienda_id, step): recalcular siempre da el mismo resultado.
    """
    from app.models.comunidad import Comunidad
    from datetime import datetime

    com = db.session.get(Comunidad, comunidad_id)
    if not com:
        return 0, 'Comunidad no encontrada'

    viviendas = Vivienda.query.filter_by(comunidad_oid=comunidad_id).all()
    if not viviendas:
        return 0, 'Sin viviendas'

    # Pesos de generación: kwp instalado (todos orientados al sur en el seed)
    gen_raw = [v.potencia_pico_paneles_kwp or 0.0 for v in viviendas]
    gen_base_total = sum(gen_raw)
    if gen_base_total <= 0:
        return 0, 'Sin potencia pico (¿ninguna vivienda con paneles?)'
    gen_pesos = [x / gen_base_total for x in gen_raw]

    # Pesos de consumo: coeficiente_reparto normalizado
    con_raw = [v.coeficiente_reparto for v in viviendas]
    con_base_total = sum(con_raw) or 1.0
    con_pesos = [x / con_base_total for x in con_raw]

    viv_ids = [v.id for v in viviendas]

    try:
        inicio = datetime.strptime(mes + '-01', '%Y-%m-%d')
        fin = datetime(inicio.year + 1, 1, 1) if inicio.month == 12 \
              else datetime(inicio.year, inicio.month + 1, 1)
    except ValueError:
        return 0, f'Formato de mes inválido: {mes} (esperado YYYY-MM)'

    ops = OperacionHoraria.query.filter(
        OperacionHoraria.comunidad_oid == comunidad_id,
        OperacionHoraria.ts >= inicio,
        OperacionHoraria.ts < fin,
    ).all()

    if not ops:
        return 0, f'Sin operaciones registradas para {mes}'

    acums = [dict(consumo=0.0, autoconsumo=0.0, bat=0.0, vertido=0.0,
                  compra_red_cost=0.0, compensacion=0.0,
                  coste_sin_paneles=0.0,
                  compra_base_cost=0.0, compensacion_base=0.0)
             for _ in viviendas]

    for op in ops:
        step   = int(op.step) if op.step is not None else op.id
        ct     = op.consumo_total_kwh or 0.0
        gt     = op.gen_total_kwh     or 0.0
        pc     = op.precio_compra     or 0.0
        pe     = op.precio_exc        or 0.0
        pd_bat = op.p_descarga_casa   or 0.0

        # Distribución individual reproducible hora a hora
        consumos = _distribuir(ct, con_pesos, step, viv_ids, _SIGMA_CONSUMO)
        gens     = _distribuir(gt, gen_pesos, step, viv_ids, _SIGMA_GEN)

        for j, viv in enumerate(viviendas):
            c = consumos[j]
            g = gens[j]
            b = pd_bat * viv.coeficiente_reparto  # batería proporcional a coef

            auto    = min(g, c)
            surplus = max(0.0, g - c)
            deficit = max(0.0, c - g)
            compra  = max(0.0, deficit - b)

            acums[j]['consumo']         += c
            acums[j]['autoconsumo']     += auto
            acums[j]['bat']             += b
            acums[j]['vertido']         += surplus
            acums[j]['compra_red_cost']   += compra * pc
            acums[j]['compensacion']      += surplus * pe
            acums[j]['coste_sin_paneles'] += c * pc
            # factura_base: escenario "paneles propios, sin batería ni comunidad"
            compra_base  = max(0.0, c - g)
            surplus_base = max(0.0, g - c)
            acums[j]['compra_base_cost']  += compra_base * pc
            acums[j]['compensacion_base'] += surplus_base * pe

    n_creados = 0
    for viv, acum in zip(viviendas, acums):
        coef = viv.coeficiente_reparto
        factura_sin_paneles = round(acum['coste_sin_paneles'], 2)
        # ahorro = valor añadido de la comunidad (batería + reparto) sobre solar individual
        factura_base = round(max(0.0, acum['compra_base_cost'] - acum['compensacion_base']), 2)
        factura_real = round(max(0.0, acum['compra_red_cost'] - acum['compensacion']), 2)
        ahorro       = round(max(0.0, factura_base - factura_real), 2)

        existente = CierreMensual.query.filter_by(vivienda_oid=viv.id, mes=mes).first()
        if existente:
            existente.consumo_total_kwh            = round(acum['consumo'], 3)
            existente.autoconsumo_directo_kwh      = round(acum['autoconsumo'], 3)
            existente.energia_de_bateria_kwh       = round(acum['bat'], 3)
            existente.vertido_a_red_kwh            = round(acum['vertido'], 3)
            existente.ahorro_eur                   = ahorro
            existente.factura_sin_paneles_eur      = factura_sin_paneles
            existente.factura_escenario_base_eur   = factura_base
            existente.factura_escenario_real_eur   = factura_real
            existente.porcentaje_ahorro_global     = round(coef * 100, 1)
            existente.coeficiente_reparto_aplicado = coef
        else:
            c_obj = CierreMensual(
                vivienda_oid=viv.id, mes=mes,
                consumo_total_kwh=round(acum['consumo'], 3),
                autoconsumo_directo_kwh=round(acum['autoconsumo'], 3),
                energia_de_bateria_kwh=round(acum['bat'], 3),
                vertido_a_red_kwh=round(acum['vertido'], 3),
                ahorro_eur=ahorro,
                factura_sin_paneles_eur=factura_sin_paneles,
                factura_escenario_base_eur=factura_base,
                factura_escenario_real_eur=factura_real,
                porcentaje_ahorro_global=round(coef * 100, 1),
                coeficiente_reparto_aplicado=coef,
            )
            db.session.add(c_obj)
            n_creados += 1

            accesos = AccesoVivienda.query.filter_by(vivienda_oid=viv.id).all()
            for a in accesos:
                if a.rol_en_vivienda in ('titular', 'convivente'):
                    crear_notificacion(
                        a.usuario_oid, 'cierre_disponible',
                        f'Cierre real de {mes} disponible',
                        f'El cierre energético de {mes} con datos reales del controlador '
                        f'ya está disponible. Ahorro: {ahorro:.2f} €',
                    )

    db.session.commit()
    total = len(viviendas)
    return total, f'{total} cierres generados/actualizados para {mes} ({len(ops)} horas)'


@cierres_bp.route('/comunidad/<comunidad_safe_oid>/generar/<mes>', methods=['POST'])
@login_required
@superadmin_required
def generar_desde_operaciones(comunidad_safe_oid, mes):
    """Genera CierreMensual para todas las viviendas a partir de OperacionHoraria."""
    try:
        com_oid = oid_from_safe(comunidad_safe_oid)
    except Exception:
        abort(404)

    n, msg = _calcular_cierre_desde_operaciones(com_oid, mes)
    if n == 0:
        flash_error(f'No se pudo generar el cierre: {msg}')
    else:
        flash_exito(msg)

    return redirect(url_for('comunidades.detalle', safe_oid=comunidad_safe_oid))


@cierres_bp.route('/vivienda/<vivienda_safe_oid>/')
@login_required
def lista(vivienda_safe_oid):
    try:
        viv_oid = oid_from_safe(vivienda_safe_oid)
    except Exception:
        abort(404)
    viv = db.session.get(Vivienda, viv_oid)
    if not viv:
        abort(404)
    _comprobar_acceso_o_superadmin(viv_oid)

    mes_filtro = request.args.get('mes', '')
    query = CierreMensual.query.filter_by(vivienda_oid=viv_oid)
    if mes_filtro:
        query = query.filter(CierreMensual.mes.contains(mes_filtro))
    cierres = query.order_by(CierreMensual.mes.desc()).all()
    from app.models.comunidad import Comunidad
    com = db.session.get(Comunidad, viv.comunidad_oid)

    return render_template('cierres/lista.html',
                           cierres=cierres, viv=viv,
                           vivienda_safe_oid=vivienda_safe_oid,
                           com=com, mes_filtro=mes_filtro,
                           es_admin=current_user.es_superadmin)


@cierres_bp.route('/vivienda/<vivienda_safe_oid>/nuevo', methods=['GET', 'POST'])
@login_required
@superadmin_required
def nuevo(vivienda_safe_oid):
    from app.models.comunidad import Comunidad
    try:
        viv_oid = oid_from_safe(vivienda_safe_oid)
    except Exception:
        abort(404)
    viv = db.session.get(Vivienda, viv_oid)
    if not viv:
        abort(404)

    form = CierreForm()
    com = db.session.get(Comunidad, viv.comunidad_oid)

    if form.validate_on_submit():
        mes = form.mes.data
        existe = CierreMensual.query.filter_by(
            vivienda_oid=viv_oid, mes=mes
        ).first()
        if existe:
            flash_error(f'Ya existe un cierre para {mes} en esta vivienda.')
            return render_template('cierres/form.html', form=form, viv=viv,
                                   com=com, vivienda_safe_oid=vivienda_safe_oid)
        cierre = CierreMensual(
            vivienda_oid=viv_oid, mes=mes,
            consumo_total_kwh=form.consumo_total_kwh.data,
            autoconsumo_directo_kwh=form.autoconsumo_directo_kwh.data,
            energia_de_bateria_kwh=form.energia_de_bateria_kwh.data,
            vertido_a_red_kwh=form.vertido_a_red_kwh.data,
            ahorro_eur=form.ahorro_eur.data,
            factura_escenario_base_eur=form.factura_escenario_base_eur.data,
            factura_escenario_real_eur=form.factura_escenario_real_eur.data,
            porcentaje_ahorro_global=form.porcentaje_ahorro_global.data,
            coeficiente_reparto_aplicado=form.coeficiente_reparto_aplicado.data
        )
        db.session.add(cierre)
        db.session.commit()

        accesos = AccesoVivienda.query.filter_by(vivienda_oid=viv_oid).all()
        for a in accesos:
            if a.rol_en_vivienda in ('titular', 'convivente'):
                crear_notificacion(
                    a.usuario_oid, 'cierre_disponible',
                    f'Cierre de {mes} disponible',
                    f'El cierre energético de {mes} de tu vivienda ya está disponible. Ahorro: {cierre.ahorro_eur:.2f} €',
                    entidad_oid=cierre.id, entidad_tipo='CierreMensual'
                )

        flash_exito(f'Cierre de {mes} creado correctamente.')
        return redirect(url_for('cierres.lista', vivienda_safe_oid=vivienda_safe_oid))
    return render_template('cierres/form.html', form=form, viv=viv,
                           com=com, vivienda_safe_oid=vivienda_safe_oid)


@cierres_bp.route('/<safe_oid>')
@login_required
def detalle(safe_oid):
    from app.models.comunidad import Comunidad
    try:
        oid = oid_from_safe(safe_oid)
    except Exception:
        abort(404)
    cierre = db.session.get(CierreMensual, oid)
    if not cierre:
        abort(404)

    viv_oid = cierre.vivienda_oid
    viv = db.session.get(Vivienda, viv_oid)
    if not viv:
        abort(404)
    _comprobar_acceso_o_superadmin(viv_oid)

    vivienda_safe_oid = oid_to_safe(viv_oid)
    com = db.session.get(Comunidad, viv.comunidad_oid)

    chart_energia = json.dumps({
        'labels': ['Autoconsumo directo', 'De batería', 'De red'],
        'datos': [
            cierre.autoconsumo_directo_kwh,
            cierre.energia_de_bateria_kwh,
            cierre.energia_de_red_kwh
        ]
    })

    return render_template('cierres/detalle.html',
                           cierre=cierre, safe_oid=safe_oid,
                           viv=viv, vivienda_safe_oid=vivienda_safe_oid,
                           com=com, es_admin=current_user.es_superadmin,
                           chart_energia=chart_energia)


@cierres_bp.route('/<safe_oid>/eliminar', methods=['POST'])
@login_required
@superadmin_required
def eliminar(safe_oid):
    try:
        oid = oid_from_safe(safe_oid)
    except Exception:
        if is_xhr():
            return jsonify({'success': False, 'error': 'OID inválida'}), 404
        abort(404)
    cierre = db.session.get(CierreMensual, oid)
    if not cierre:
        if is_xhr():
            return jsonify({'success': False, 'error': 'No encontrado'}), 404
        abort(404)

    vivienda_safe_oid = oid_to_safe(cierre.vivienda_oid)
    mes = cierre.mes
    db.session.delete(cierre)
    db.session.commit()
    flash_exito(f'Cierre de {mes} eliminado.')
    if is_xhr():
        return jsonify({'success': True})
    return redirect(url_for('cierres.lista', vivienda_safe_oid=vivienda_safe_oid))
