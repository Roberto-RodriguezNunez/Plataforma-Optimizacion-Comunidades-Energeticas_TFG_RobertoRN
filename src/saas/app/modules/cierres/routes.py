"""CRUD de CierresMensuales."""
from flask import render_template, redirect, url_for, request, abort, jsonify
from flask_login import login_required, current_user
import json

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


def _comprobar_acceso_o_superadmin(viv_oid):
    if current_user.es_superadmin:
        return
    if not usuario_tiene_acceso(current_user.id, viv_oid):
        abort(403)


def _calcular_cierre_desde_operaciones(comunidad_id, mes):
    """Genera o actualiza CierreMensual para cada vivienda a partir de OperacionHoraria.

    Fórmula (RD 244/2019 autoconsumo colectivo):
      consumo_i      = consumo_total × coef_reparto_i
      gen_atrib_i    = gen_total × (viv.kwp / total_kwp)
      bat_cubierto_i = p_descarga_casa × coef_reparto_i
      compra_red_i   = max(0, deficit_i - bat_cubierto_i)
      surplus_i      = max(0, gen_atrib_i - consumo_i)
      ahorro_i       = coste_base_i - (compra_red_i × pc - surplus_i × pe)
    """
    from app.models.comunidad import Comunidad
    from datetime import datetime

    com = db.session.get(Comunidad, comunidad_id)
    if not com:
        return 0, 'Comunidad no encontrada'

    viviendas = Vivienda.query.filter_by(comunidad_oid=comunidad_id).all()
    if not viviendas:
        return 0, 'Sin viviendas'

    total_kwp = sum(v.potencia_pico_paneles_kwp or 0.0 for v in viviendas)
    if total_kwp <= 0:
        return 0, 'Sin potencia pico total (¿alguna vivienda sin paneles?)'

    # Operaciones del mes pedido (YYYY-MM)
    try:
        inicio = datetime.strptime(mes + '-01', '%Y-%m-%d')
        if inicio.month == 12:
            fin = datetime(inicio.year + 1, 1, 1)
        else:
            fin = datetime(inicio.year, inicio.month + 1, 1)
    except ValueError:
        return 0, f'Formato de mes inválido: {mes} (esperado YYYY-MM)'

    ops = OperacionHoraria.query.filter(
        OperacionHoraria.comunidad_oid == comunidad_id,
        OperacionHoraria.ts >= inicio,
        OperacionHoraria.ts < fin,
    ).all()

    if not ops:
        return 0, f'Sin operaciones registradas para {mes}'

    n_creados = 0
    for viv in viviendas:
        coef = viv.coeficiente_reparto
        kwp  = viv.potencia_pico_paneles_kwp or 0.0
        frac_gen = kwp / total_kwp

        acum = dict(consumo=0.0, autoconsumo=0.0, bat=0.0,
                    vertido=0.0, compra_red_cost=0.0, compensacion=0.0,
                    coste_base=0.0)

        for op in ops:
            c = (op.consumo_total_kwh or 0.0) * coef
            g = (op.gen_total_kwh or 0.0) * frac_gen
            b = (op.p_descarga_casa or 0.0) * coef
            pc = op.precio_compra or 0.0
            pe = op.precio_exc or 0.0

            auto    = min(g, c)
            surplus = max(0.0, g - c)
            deficit = max(0.0, c - g)
            compra  = max(0.0, deficit - b)

            acum['consumo']         += c
            acum['autoconsumo']     += auto
            acum['bat']             += b
            acum['vertido']         += surplus
            acum['compra_red_cost'] += compra * pc
            acum['compensacion']    += surplus * pe
            acum['coste_base']      += c * pc

        factura_base = round(acum['coste_base'], 2)
        factura_real = round(
            max(0.0, acum['compra_red_cost'] - acum['compensacion']), 2
        )
        ahorro = round(max(0.0, factura_base - factura_real), 2)

        existente = CierreMensual.query.filter_by(
            vivienda_oid=viv.id, mes=mes
        ).first()
        if existente:
            existente.consumo_total_kwh        = round(acum['consumo'], 3)
            existente.autoconsumo_directo_kwh  = round(acum['autoconsumo'], 3)
            existente.energia_de_bateria_kwh   = round(acum['bat'], 3)
            existente.vertido_a_red_kwh        = round(acum['vertido'], 3)
            existente.ahorro_eur               = ahorro
            existente.factura_escenario_base_eur = factura_base
            existente.factura_escenario_real_eur = factura_real
            existente.porcentaje_ahorro_global = round(coef * 100, 1)
            existente.coeficiente_reparto_aplicado = coef
        else:
            c_obj = CierreMensual(
                vivienda_oid=viv.id, mes=mes,
                consumo_total_kwh=round(acum['consumo'], 3),
                autoconsumo_directo_kwh=round(acum['autoconsumo'], 3),
                energia_de_bateria_kwh=round(acum['bat'], 3),
                vertido_a_red_kwh=round(acum['vertido'], 3),
                ahorro_eur=ahorro,
                factura_escenario_base_eur=factura_base,
                factura_escenario_real_eur=factura_real,
                porcentaje_ahorro_global=round(coef * 100, 1),
                coeficiente_reparto_aplicado=coef,
            )
            db.session.add(c_obj)
            n_creados += 1

            # Notificar titulares y conviventes
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
