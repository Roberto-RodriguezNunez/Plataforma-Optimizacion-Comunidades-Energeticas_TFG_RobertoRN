"""CRUD de Viviendas."""
import json
from flask import render_template, redirect, url_for, request, abort, jsonify
from flask_login import login_required, current_user
from datetime import date

from app.extensions import db
from app.modules.viviendas import viviendas_bp
from app.modules.viviendas.forms import ViviendaForm
from app.models.vivienda import Vivienda
from app.models.comunidad import Comunidad
from app.models.cierre import CierreMensual
from app.helpers import (oid_from_safe, oid_to_safe, flash_exito, flash_error,
                          is_xhr, cascade_delete_vivienda, usuario_tiene_acceso,
                          recalcular_coeficientes)
from app.decorators import superadmin_required


def _get_vivienda_o_404(safe_oid):
    try:
        oid = oid_from_safe(safe_oid)
    except Exception:
        abort(404)
    viv = db.session.get(Vivienda, oid)
    if not viv:
        abort(404)
    return oid, viv


def _comprobar_acceso_vivienda(viv, oid):
    if current_user.es_superadmin:
        return
    if not usuario_tiene_acceso(current_user.id, oid):
        abort(403)


@viviendas_bp.route('/')
@login_required
@superadmin_required
def lista_global():
    todas = Vivienda.query.all()
    comunidades = {str(c.id): c for c in Comunidad.query.all()}
    # Agrupar por comunidad manteniendo orden
    grupos = {}
    for v in todas:
        key = str(v.comunidad_oid)
        grupos.setdefault(key, []).append(v)
    grupos_lista = [
        {'com': comunidades.get(k), 'com_str': k, 'viviendas': vs}
        for k, vs in grupos.items()
    ]
    return render_template('viviendas/lista_global.html',
                           grupos=grupos_lista,
                           total=len(todas))


@viviendas_bp.route('/comunidad/<comunidad_safe_oid>/')
@login_required
def lista(comunidad_safe_oid):
    try:
        com_oid = oid_from_safe(comunidad_safe_oid)
    except Exception:
        abort(404)
    com = db.session.get(Comunidad, com_oid)
    if not com:
        abort(404)

    viviendas = Vivienda.query.filter_by(comunidad_oid=com_oid).all()

    if not current_user.es_superadmin:
        from app.models.acceso import AccesoVivienda
        viv_ids = [v.id for v in viviendas]
        acceso = None
        if viv_ids:
            acceso = AccesoVivienda.query.filter(
                AccesoVivienda.usuario_oid == current_user.id,
                AccesoVivienda.vivienda_oid.in_(viv_ids),
            ).first()
        if not acceso:
            abort(403)

    return render_template('viviendas/lista.html', viviendas=viviendas,
                           com=com, comunidad_safe_oid=comunidad_safe_oid)


@viviendas_bp.route('/comunidad/<comunidad_safe_oid>/nueva', methods=['GET', 'POST'])
@login_required
@superadmin_required
def nueva(comunidad_safe_oid):
    try:
        com_oid = oid_from_safe(comunidad_safe_oid)
    except Exception:
        abort(404)
    com = db.session.get(Comunidad, com_oid)
    if not com:
        abort(404)

    form = ViviendaForm()
    if form.validate_on_submit():
        existe = Vivienda.query.filter_by(
            comunidad_oid=com_oid, identificador=form.identificador.data
        ).first()
        if existe:
            flash_error('Ya existe una vivienda con ese identificador en esta comunidad.')
            return render_template('viviendas/form.html', form=form,
                                   titulo='Nueva vivienda', com=com,
                                   comunidad_safe_oid=comunidad_safe_oid)
        viv = Vivienda(
            comunidad_oid=com_oid,
            identificador=form.identificador.data,
            direccion_completa=form.direccion_completa.data or '',
            cups=form.cups.data or '',
            potencia_contratada_kw=form.potencia_contratada_kw.data,
            coeficiente_reparto=0.0,
            fecha_alta=form.fecha_alta.data.isoformat(),
            tiene_paneles=form.tiene_paneles.data,
            potencia_pico_paneles_kwp=form.potencia_pico_paneles_kwp.data if form.tiene_paneles.data else None,
            numero_paneles=form.numero_paneles.data if form.tiene_paneles.data else None,
            fecha_instalacion_paneles=form.fecha_instalacion_paneles.data.isoformat() if form.tiene_paneles.data and form.fecha_instalacion_paneles.data else None,
            orientacion_paneles=form.orientacion_paneles.data if form.tiene_paneles.data else None
        )
        db.session.add(viv)
        db.session.commit()
        recalcular_coeficientes(com_oid)
        flash_exito(f'Vivienda "{viv.identificador}" creada. Coeficientes recalculados.')
        return redirect(url_for('comunidades.detalle', safe_oid=comunidad_safe_oid))
    return render_template('viviendas/form.html', form=form,
                           titulo='Nueva vivienda', com=com,
                           comunidad_safe_oid=comunidad_safe_oid)


@viviendas_bp.route('/<safe_oid>')
@login_required
def detalle(safe_oid):
    oid, viv = _get_vivienda_o_404(safe_oid)
    _comprobar_acceso_vivienda(viv, oid)

    com_oid = viv.comunidad_oid
    com = db.session.get(Comunidad, com_oid)
    com_safe_oid = oid_to_safe(com_oid)

    cierres = CierreMensual.query.filter_by(
        vivienda_oid=oid
    ).order_by(CierreMensual.mes).all()

    # ----------------------------------------------------------------
    # Gráfica 1: Ahorro con comunidad vs los 2 baselines — 2 líneas
    # ----------------------------------------------------------------
    grafica = cierres[-12:]
    serie_sin = [round(max(0.0, c.factura_sin_paneles_eur - c.factura_escenario_real_eur), 2) for c in grafica]
    serie_solo = [round(max(0.0, c.factura_escenario_base_eur - c.factura_escenario_real_eur), 2) for c in grafica]
    chart_ahorro = json.dumps({
        'labels': [c.mes for c in grafica],
        'vs_sin_paneles': serie_sin,
        'vs_solo_paneles': serie_solo,
    })
    ahorros_totales = {
        'vs_sin_paneles':  round(sum(serie_sin), 2),
        'vs_solo_paneles': round(sum(serie_solo), 2),
    }

    # ----------------------------------------------------------------
    # Gráfica 2: Comparativa de 3 escenarios — barras agrupadas
    # ----------------------------------------------------------------
    chart_compare = json.dumps({
        'labels':             [c.mes for c in grafica],
        'sin_paneles':        [round(c.factura_sin_paneles_eur, 2) for c in grafica],
        'solo_paneles':       [round(c.factura_escenario_base_eur, 2) for c in grafica],
        'comunidad_completa': [round(c.factura_escenario_real_eur, 2) for c in grafica],
    })

    # ----------------------------------------------------------------
    # Gráfica 3: Mix energético del último mes — doughnut
    # ----------------------------------------------------------------
    if cierres:
        ult = cierres[-1]
        chart_mix = json.dumps({
            'labels': ['Autoconsumo directo', 'Autoconsumo indirecto (comunidad)',
                       'Batería comunitaria', 'De la red'],
            'datos':  [ult.autoconsumo_directo_kwh,
                       round(ult.autoconsumo_indirecto_kwh, 2),
                       ult.energia_de_bateria_kwh,
                       round(ult.energia_de_red_kwh, 2)],
            'mes': ult.mes
        })
    else:
        chart_mix = None

    # ----------------------------------------------------------------
    # Gráfica 4: Comparativa con media de la comunidad — barras horizontales
    # Calculamos el ahorro medio mensual de cada vivienda de la comunidad
    # ----------------------------------------------------------------
    viviendas_com = Vivienda.query.filter_by(comunidad_oid=com_oid).all()
    community_bars = []
    for v in viviendas_com:
        cierres_v = CierreMensual.query.filter_by(vivienda_oid=v.id).all()
        if cierres_v:
            media = round(sum(c.ahorro_eur for c in cierres_v) / len(cierres_v), 2)
            community_bars.append({
                'label': v.identificador,
                'media': media,
                'es_esta': v.id == oid
            })
    community_bars.sort(key=lambda x: x['media'], reverse=True)
    chart_community = json.dumps({
        'labels':   [b['label'] for b in community_bars],
        'medias':   [b['media'] for b in community_bars],
        'highlight': next((i for i, b in enumerate(community_bars) if b['es_esta']), None),
    }) if community_bars else None

    # ----------------------------------------------------------------
    # KPIs
    # ----------------------------------------------------------------
    ahorro_total  = round(sum(c.ahorro_eur for c in cierres), 2)
    ahorro_medio  = round(ahorro_total / len(cierres), 2) if cierres else 0.0
    pct_ahorro    = round(
        sum(c.ahorro_eur for c in cierres) /
        sum(c.factura_escenario_base_eur for c in cierres) * 100
        if cierres and sum(c.factura_escenario_base_eur for c in cierres) > 0 else 0, 1)

    return render_template('viviendas/detalle.html',
                           viv=viv, safe_oid=safe_oid,
                           com=com, com_safe_oid=com_safe_oid,
                           cierres=cierres,
                           cierres_tabla=cierres[:6],
                           n_cierres=len(cierres),
                           es_admin=current_user.es_superadmin,
                           chart_ahorro=chart_ahorro,
                           ahorros_totales=ahorros_totales,
                           chart_compare=chart_compare,
                           chart_mix=chart_mix,
                           chart_community=chart_community,
                           ahorro_total=ahorro_total,
                           ahorro_medio=ahorro_medio,
                           pct_ahorro=pct_ahorro)


@viviendas_bp.route('/<safe_oid>/editar', methods=['GET', 'POST'])
@login_required
@superadmin_required
def editar(safe_oid):
    oid, viv = _get_vivienda_o_404(safe_oid)
    com_oid = viv.comunidad_oid
    com = db.session.get(Comunidad, com_oid)
    comunidad_safe_oid = oid_to_safe(com_oid)

    form = ViviendaForm(obj=viv)
    if request.method == 'GET':
        try:
            form.fecha_alta.data = date.fromisoformat(viv.fecha_alta)
            if viv.fecha_instalacion_paneles:
                form.fecha_instalacion_paneles.data = date.fromisoformat(viv.fecha_instalacion_paneles)
        except Exception:
            pass

    if form.validate_on_submit():
        duplicada = Vivienda.query.filter(
            Vivienda.comunidad_oid == com_oid,
            Vivienda.identificador == form.identificador.data,
            Vivienda.id != oid,
        ).first()
        if duplicada:
            flash_error('Ya existe otra vivienda con ese identificador en esta comunidad.')
            return render_template('viviendas/form.html', form=form,
                                   titulo='Editar vivienda', com=com,
                                   safe_oid=safe_oid, comunidad_safe_oid=comunidad_safe_oid)
        kwp_antes = viv.potencia_pico_paneles_kwp or 0.0
        kwp_nuevo = (form.potencia_pico_paneles_kwp.data or 0.0) if form.tiene_paneles.data else 0.0
        paneles_cambio = kwp_antes != kwp_nuevo
        viv.identificador = form.identificador.data
        viv.direccion_completa = form.direccion_completa.data or ''
        viv.cups = form.cups.data or ''
        viv.potencia_contratada_kw = form.potencia_contratada_kw.data
        viv.fecha_alta = form.fecha_alta.data.isoformat()
        viv.tiene_paneles = form.tiene_paneles.data
        if viv.tiene_paneles:
            viv.potencia_pico_paneles_kwp = form.potencia_pico_paneles_kwp.data
            viv.numero_paneles = form.numero_paneles.data
            viv.fecha_instalacion_paneles = form.fecha_instalacion_paneles.data.isoformat() if form.fecha_instalacion_paneles.data else None
            viv.orientacion_paneles = form.orientacion_paneles.data
        else:
            viv.potencia_pico_paneles_kwp = None
            viv.numero_paneles = None
            viv.fecha_instalacion_paneles = None
            viv.orientacion_paneles = None
        db.session.commit()
        if paneles_cambio:
            recalcular_coeficientes(com_oid)
        flash_exito(f'Vivienda "{viv.identificador}" actualizada.')
        return redirect(url_for('viviendas.detalle', safe_oid=safe_oid))
    return render_template('viviendas/form.html', form=form,
                           titulo='Editar vivienda', com=com,
                           safe_oid=safe_oid, comunidad_safe_oid=comunidad_safe_oid)


@viviendas_bp.route('/<safe_oid>/eliminar', methods=['POST'])
@login_required
@superadmin_required
def eliminar(safe_oid):
    try:
        oid = oid_from_safe(safe_oid)
    except Exception:
        if is_xhr():
            return jsonify({'success': False, 'error': 'OID inválida'}), 404
        abort(404)
    viv = db.session.get(Vivienda, oid)
    if not viv:
        if is_xhr():
            return jsonify({'success': False, 'error': 'No encontrada'}), 404
        abort(404)

    com_oid = viv.comunidad_oid
    com_safe_oid = oid_to_safe(com_oid)
    nombre = viv.identificador
    cascade_delete_vivienda(oid)
    recalcular_coeficientes(com_oid)
    flash_exito(f'Vivienda "{nombre}" eliminada. Coeficientes recalculados.')
    if is_xhr():
        return jsonify({'success': True, 'redirect': url_for('comunidades.detalle', safe_oid=com_safe_oid)})
    return redirect(url_for('comunidades.detalle', safe_oid=com_safe_oid))
