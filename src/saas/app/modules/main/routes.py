"""Rutas del panel principal (dashboard) según rol del usuario."""
import json
from collections import defaultdict
from flask import render_template
from flask_login import login_required, current_user

from app.extensions import db
from app.modules.main import main_bp
from app.models.comunidad import Comunidad
from app.models.vivienda import Vivienda
from app.models.usuario import Usuario
from app.models.acceso import AccesoVivienda
from app.models.bateria import Bateria
from app.models.cierre import CierreMensual
from app.helpers import oid_to_safe


@main_bp.route('/')
def index():
    if not current_user.is_authenticated:
        return render_template('main/landing.html')

    if current_user.es_superadmin:
        comunidades = Comunidad.query.all()
        todas_viviendas = Vivienda.query.all()
        # Agrupar viviendas por comunidad
        viviendas_por_com = {}
        for v in todas_viviendas:
            key = str(v.comunidad_oid)
            viviendas_por_com.setdefault(key, []).append(v)
        return render_template('main/dashboard_superadmin.html',
                               comunidades=comunidades,
                               viviendas_por_com=viviendas_por_com,
                               total_viviendas=len(todas_viviendas),
                               total_usuarios=Usuario.query.count(),
                               total_comunidades=len(comunidades))

    # ── Usuario normal ──────────────────────────────────────────────
    accesos = AccesoVivienda.query.filter_by(usuario_oid=current_user.id).all()

    # Cache de comunidades ya cargadas
    com_cache = {}

    viviendas_data = []
    ahorro_global  = 0.0

    for a in accesos:
        try:
            viv = db.session.get(Vivienda, a.vivienda_oid)
            if not viv:
                continue

            # Comunidad de esta vivienda
            com_str = str(viv.comunidad_oid)
            if com_str not in com_cache:
                com_cache[com_str] = db.session.get(Comunidad, viv.comunidad_oid)
            com = com_cache[com_str]

            # Batería de la comunidad
            bat = Bateria.query.filter_by(comunidad_oid=viv.comunidad_oid).first()
            bat_pct = None
            if bat:
                bat_pct = min(round(bat.capacidad_util_actual_kwh
                                    / bat.capacidad_nominal_kwh * 100), 100)

            # Cierres de esta vivienda
            cierres = CierreMensual.query.filter_by(
                vivienda_oid=viv.id
            ).order_by(CierreMensual.mes).all()

            ahorro_total = round(sum(c.ahorro_eur for c in cierres), 2)
            ahorro_global += ahorro_total

            if cierres:
                ult = cierres[-1]
                ultimo_ahorro = round(ult.ahorro_eur, 2)
                ultimo_pct    = round(
                    (ult.ahorro_eur / ult.factura_escenario_base_eur * 100)
                    if ult.factura_escenario_base_eur else 0, 1)
                ultimo_mes = ult.mes
                # Gráfica ahorro mensual (últimos 6)
                ultimos = cierres[-6:]
                chart_ahorro = json.dumps({
                    'labels': [c.mes for c in ultimos],
                    'ahorro': [round(c.ahorro_eur, 2) for c in ultimos],
                    'base':   [round(c.factura_escenario_base_eur, 2) for c in ultimos],
                    'real':   [round(c.factura_escenario_real_eur, 2) for c in ultimos],
                })
            else:
                ultimo_ahorro = 0.0
                ultimo_pct    = 0.0
                ultimo_mes    = '—'
                chart_ahorro  = None

            viviendas_data.append({
                'viv':          viv,
                'acceso':       a,
                'safe_oid':     oid_to_safe(viv.__oid__),
                'com':          com,
                'com_safe_oid': oid_to_safe(com.__oid__) if com else None,
                'bat_pct':      bat_pct,
                'ahorro_total': ahorro_total,
                'ultimo_ahorro':ultimo_ahorro,
                'ultimo_pct':   ultimo_pct,
                'ultimo_mes':   ultimo_mes,
                'n_cierres':    len(cierres),
                'chart_ahorro': chart_ahorro,
            })
        except Exception:
            pass

    # Incidencias abiertas del usuario
    from app.models.incidencia import Incidencia
    incidencias_abiertas = Incidencia.query.filter(
        Incidencia.usuario_oid == current_user.id,
        Incidencia.estado != 'cerrada',
    ).all()

    return render_template('main/dashboard_usuario.html',
                           viviendas_data=viviendas_data,
                           ahorro_global=round(ahorro_global, 2),
                           incidencias_abiertas=incidencias_abiertas)
