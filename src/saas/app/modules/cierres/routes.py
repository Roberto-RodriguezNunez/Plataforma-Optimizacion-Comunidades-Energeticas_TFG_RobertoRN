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
# σ_con=0.35: CV≈36 %, acorde con la heterogeneidad real de consumo doméstico
#   (diferencias de ocupación, horarios, electrodomésticos, VE…)
# σ_gen=0.10: CV≈10 %, varianza intra-instalación con orientación uniforme
#   (ensuciamiento diferencial, microsombras locales, eficiencia de inversor/cableado)
_SIGMA_CONSUMO = 0.35
_SIGMA_GEN     = 0.10


def _distribuir(total, pesos_norm, step, vivienda_ids, sigma, salt=0):
    """Distribuye `total` entre N viviendas con ruido log-normal reproducible.

    Cada casa recibe un peso base (coef o kwp×orient) perturbado con ruido
    log-normal de desviación `sigma`, sembrado deterministicamente por
    (vivienda_id, step) para que recalcular el cierre dé siempre el mismo
    resultado. La normalización garantiza sum(devuelto) == total exactamente.

    `salt` separa el flujo de ruido de consumo del de generación: con la misma
    semilla, el shock z de consumo y el de generación serían idénticos (una casa
    que por azar consume más generaría más en lockstep), lo que anula la
    diversidad de perfiles de la que vive el autoconsumo colectivo. Con salt
    distinto los dos shocks son independientes, como en la realidad.
    """
    p = list(pesos_norm)
    for i, vid in enumerate(vivienda_ids):
        r = _rng.Random(int(vid) * 99991 + int(step) + salt)
        # Box-Muller: N(0,1) sin dependencias externas
        u1, u2 = max(r.random(), 1e-12), r.random()
        z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        p[i] = max(p[i] * math.exp(sigma * z), 1e-9)
    s = sum(p)
    return [total * x / s for x in p]


def _repartir_ahorro_sin_negativos(coefs, facturas_base, ahorro_total):
    """Reparte `ahorro_total` entre viviendas proporcionalmente a su coeficiente
    (kWp aportado), pero capando el ahorro de cada vivienda a su propia factura
    base para que ninguna cuota resulte negativa: la cooperativa reduce el gasto
    de cada socio, nunca le paga. El exceso de las viviendas cuya parte
    proporcional supera su factura base se redistribuye entre las demás
    (water-filling).

    Devuelve la lista de ahorros con `0 <= ahorro_i <= facturas_base_i`. Como
    `ahorro_total <= sum(facturas_base)` por construcción (ahorro = max(0,
    Σbase − factura_común) y la factura común es ≥ 0), se cumple además
    `sum(ahorros) == ahorro_total`, de modo que Σ cuotas == factura común. Cuando
    ninguna vivienda se satura, `ahorro_i == coef_i * ahorro_total` exactamente.
    """
    n = len(coefs)
    ahorros = [0.0] * n
    activo = [True] * n
    for _ in range(n + 1):
        coef_activo = sum(coefs[i] for i in range(n) if activo[i])
        restante = ahorro_total - sum(ahorros)
        if restante <= 1e-12 or coef_activo <= 1e-12:
            break
        # ¿Alguna activa superaría su factura base con el reparto proporcional?
        capadas = [i for i in range(n) if activo[i] and
                   restante * coefs[i] / coef_activo > facturas_base[i] - ahorros[i]]
        if capadas:
            for i in capadas:  # satúralas a su tope y redistribuye en la próxima ronda
                ahorros[i] = facturas_base[i]
                activo[i] = False
        else:  # nadie se satura: reparto proporcional final entre las activas
            for i in range(n):
                if activo[i]:
                    ahorros[i] += restante * coefs[i] / coef_activo
            break
    return ahorros


def _comprobar_acceso_o_superadmin(viv_oid):
    if current_user.es_superadmin:
        return
    if not usuario_tiene_acceso(current_user.id, viv_oid):
        abort(403)


def _calcular_cierre_desde_operaciones(comunidad_id, mes):
    """Genera o actualiza CierreMensual para cada vivienda a partir de OperacionHoraria.

    MODELO: cooperativa con un único punto de suministro (CUPS único), modalidad de
    autoconsumo individual del RD 244/2019. La producción se NETEA hora a hora sobre
    el balance agregado de la comunidad: ese balance neto (con la batería SAC) ES la
    ÚNICA factura eléctrica oficial. Lo que se reparte entre las viviendas no es
    energía sino el AHORRO económico resultante, mediante coeficientes de reducción
    de gasto proporcionales a la producción solar estimada que aporta cada vivienda
    (kWp instalado); no es el coeficiente reglamentario del colectivo (Ley 24/2013,
    sin reventa de energía).

    Cálculo:

      factura_real_com = Σ_h [ (max(0, ct−gt−p_descarga_casa) + p_carga_red)×pc
                               − (max(0, gt−ct−p_carga_solar) + p_descarga_red)×pe ]

      consumo_i  — distribuido desde consumo_total con ruido log-normal (σ=0.35)
                   ponderado por potencia contratada; sum(consumo_i) == consumo_total.
      gen_i      — distribuido desde gen_total con ruido log-normal (σ=0.10)
                   ponderado por kwp instalado; sum(gen_i) == gen_total exacto.

      factura_sin_paneles_i = Σ consumo_i×pc          (sin paneles solares)
      factura_base_i        = Σ max(0,c−g)×pc − max(0,g−c)×pe   (sin comunidad ni batería)

      ahorro_total   = max(0, Σ_i factura_base_i − factura_real_com)
      ahorro_i       = reparto de ahorro_total ∝ coef_i, capado a factura_base_i
                       (water-filling: el exceso se redistribuye al resto), de
                       modo que 0 ≤ ahorro_i ≤ factura_base_i
      cuota_real_i   = factura_base_i − ahorro_i ≥ 0  (Σ cuotas == factura_real_com)

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

    # Pesos de consumo: potencia contratada normalizada (el coeficiente de
    # reparto ya no pondera consumo: reparte el ahorro, y va por kWp)
    con_raw = [v.potencia_contratada_kw or 0.0 for v in viviendas]
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

    acums = [dict(consumo=0.0, autoconsumo=0.0, indirecto=0.0, bat=0.0, vertido=0.0,
                  coste_sin_paneles=0.0,
                  compra_base_cost=0.0, compensacion_base=0.0)
             for _ in viviendas]
    factura_real_com = 0.0  # factura oficial: neteo horario agregado con batería

    for op in ops:
        step   = int(op.step) if op.step is not None else op.id
        ct     = op.consumo_total_kwh or 0.0
        gt     = op.gen_total_kwh     or 0.0
        pc     = op.precio_compra     or 0.0
        pe     = op.precio_exc        or 0.0
        pd_bat = op.p_descarga_casa   or 0.0

        # Neteo horario del CUPS único (con los flujos de la batería SAC)
        compra_com  = max(0.0, ct - gt - pd_bat) + (op.p_carga_red or 0.0)
        vertido_com = max(0.0, gt - ct - (op.p_carga_solar or 0.0)) + (op.p_descarga_red or 0.0)
        factura_real_com += compra_com * pc - vertido_com * pe

        # Distribución individual reproducible hora a hora.
        # salt distinto en generación → ruido de gen y consumo independientes.
        consumos = _distribuir(ct, con_pesos, step, viv_ids, _SIGMA_CONSUMO, salt=0)
        gens     = _distribuir(gt, gen_pesos, step, viv_ids, _SIGMA_GEN, salt=2_000_000)

        # Cascada horaria del origen del consumo, coherente con el neteo del CUPS
        # único: 1) sol propio (directo), 2) sol de otras viviendas que cubre el
        # déficit vía neteo (indirecto = compartido), 3) batería comunitaria
        # repartida por déficit restante (nunca más de lo que falta), 4) red.
        # Por construcción directo+indirecto+batería+red == consumo cada hora.
        directos = [min(g, c) for g, c in zip(gens, consumos)]
        deficits = [c - d for c, d in zip(consumos, directos)]        # = max(0, c-g)
        surplus  = [max(0.0, g - c) for g, c in zip(gens, consumos)]
        shared   = min(sum(surplus), sum(deficits))                   # sol compartido en la comunidad
        sum_def  = sum(deficits)
        indirectos = [d * shared / sum_def if sum_def > 1e-12 else 0.0 for d in deficits]
        rem_def  = [d - ind for d, ind in zip(deficits, indirectos)]  # déficit tras el sol
        sum_rem  = sum(rem_def)
        bat_alloc = min(pd_bat, sum_rem)                              # batería no cubre más que el déficit
        baterias = [r * bat_alloc / sum_rem if sum_rem > 1e-12 else 0.0 for r in rem_def]

        for j in range(len(viviendas)):
            acums[j]['consumo']     += consumos[j]
            acums[j]['autoconsumo'] += directos[j]
            acums[j]['indirecto']   += indirectos[j]
            acums[j]['bat']         += baterias[j]
            acums[j]['vertido']     += surplus[j]
            # Escenario sin paneles solares
            acums[j]['coste_sin_paneles'] += consumos[j] * pc
            # Escenario sin comunidad: solo sus paneles, sin batería ni reparto
            acums[j]['compra_base_cost']  += deficits[j] * pc
            acums[j]['compensacion_base'] += surplus[j] * pe

    # Ahorro total de la comunidad: lo que la cooperativa (neteo + batería)
    # ahorra frente a que cada casa fuera sola con sus paneles.
    facturas_base = [max(0.0, a['compra_base_cost'] - a['compensacion_base']) for a in acums]
    ahorro_total  = max(0.0, sum(facturas_base) - factura_real_com)

    # Reparto del ahorro por coeficiente, capado a la factura base de cada
    # vivienda (water-filling) para que ninguna cuota quede negativa.
    coefs   = [v.coeficiente_reparto for v in viviendas]
    ahorros = _repartir_ahorro_sin_negativos(coefs, facturas_base, ahorro_total)

    n_creados = 0
    for viv, acum, f_base, ahorro_viv in zip(viviendas, acums, facturas_base, ahorros):
        coef = viv.coeficiente_reparto
        factura_sin_paneles = round(acum['coste_sin_paneles'], 2)
        factura_base        = round(f_base, 2)
        # El ahorro común se resta a la factura de cada vecino; el reparto va por
        # coeficiente (kWp aportado) pero acotado a su factura base, de modo que
        # la suma de cuotas reproduce la factura común sin cuotas negativas.
        ahorro              = round(ahorro_viv, 2)
        factura_real        = round(f_base - ahorro_viv, 2)

        existente = CierreMensual.query.filter_by(vivienda_oid=viv.id, mes=mes).first()
        if existente:
            existente.consumo_total_kwh               = round(acum['consumo'], 3)
            existente.autoconsumo_directo_kwh         = round(acum['autoconsumo'], 3)
            existente.autoconsumo_indirecto_kwh       = round(acum['indirecto'], 3)
            existente.energia_de_bateria_kwh          = round(acum['bat'], 3)
            existente.vertido_a_red_kwh               = round(acum['vertido'], 3)
            existente.ahorro_eur                      = ahorro
            existente.factura_sin_paneles_eur         = factura_sin_paneles
            existente.factura_escenario_base_eur      = factura_base
            existente.factura_escenario_real_eur      = factura_real
            existente.porcentaje_ahorro_global        = round(coef * 100, 1)
            existente.coeficiente_reparto_aplicado    = coef
        else:
            c_obj = CierreMensual(
                vivienda_oid=viv.id, mes=mes,
                consumo_total_kwh=round(acum['consumo'], 3),
                autoconsumo_directo_kwh=round(acum['autoconsumo'], 3),
                autoconsumo_indirecto_kwh=round(acum['indirecto'], 3),
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
            autoconsumo_indirecto_kwh=form.autoconsumo_indirecto_kwh.data,
            energia_de_bateria_kwh=form.energia_de_bateria_kwh.data,
            vertido_a_red_kwh=form.vertido_a_red_kwh.data,
            ahorro_eur=form.ahorro_eur.data,
            factura_sin_paneles_eur=form.factura_sin_paneles_eur.data,
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
