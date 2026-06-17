"""Gestión de Incidencias."""
from datetime import date
from flask import render_template, redirect, url_for, request, abort, jsonify
from flask_login import login_required, current_user

from app.extensions import db
from app.modules.incidencias import incidencias_bp
from app.modules.incidencias.forms import IncidenciaForm, RespuestaForm
from app.models.incidencia import Incidencia
from app.models.acceso import AccesoVivienda
from app.models.vivienda import Vivienda
from app.helpers import oid_from_safe, oid_to_safe, flash_exito, flash_error, is_xhr
from app.decorators import superadmin_required


# ---------------------------------------------------------------------------
# Lista de incidencias
# ---------------------------------------------------------------------------

@incidencias_bp.route('/')
@login_required
def lista():
    if current_user.es_superadmin:
        incidencias = Incidencia.query.all()
    else:
        # Usuario normal: solo ve sus propias incidencias
        incidencias = Incidencia.query.filter_by(usuario_oid=current_user.id).all()

    # Enriquecer con datos de vivienda para la tabla
    detalle = []
    for inc in incidencias:
        viv = db.session.get(Vivienda, inc.vivienda_oid)
        detalle.append({
            'inc': inc,
            'safe_oid': oid_to_safe(inc.__oid__),
            'viv': viv,
        })

    # Ordenar: abiertas primero, luego por fecha descendente
    orden_estado = {'abierta': 0, 'en_proceso': 1, 'cerrada': 2}
    detalle.sort(key=lambda d: (
        orden_estado.get(d['inc'].estado, 9),
        d['inc'].fecha_creacion or ''
    ), reverse=False)

    return render_template('incidencias/lista.html', detalle=detalle)


# ---------------------------------------------------------------------------
# Nueva incidencia
# ---------------------------------------------------------------------------

@incidencias_bp.route('/nueva', methods=['GET', 'POST'])
@login_required
def nueva():
    if current_user.es_superadmin:
        flash_error('Los superadmin no pueden crear incidencias.')
        return redirect(url_for('incidencias.lista'))

    # Viviendas a las que tiene acceso el usuario
    accesos = AccesoVivienda.query.filter_by(usuario_oid=current_user.id).all()

    viviendas = []
    for a in accesos:
        viv = db.session.get(Vivienda, a.vivienda_oid)
        if viv:
            viviendas.append(viv)

    if not viviendas:
        flash_error('No tienes ninguna vivienda asignada. Contacta con el administrador.')
        return redirect(url_for('incidencias.lista'))

    form = IncidenciaForm()

    # Construir opciones de vivienda para el template
    viviendas_opciones = [
        (oid_to_safe(v.__oid__), v.identificador)
        for v in viviendas
    ]

    if form.validate_on_submit():
        # Determinar vivienda seleccionada
        vivienda_safe = request.form.get('vivienda_safe_oid')
        if not vivienda_safe and len(viviendas) == 1:
            vivienda_safe = oid_to_safe(viviendas[0].__oid__)

        if not vivienda_safe:
            flash_error('Debes seleccionar una vivienda.')
            return render_template('incidencias/form.html', form=form,
                                   viviendas_opciones=viviendas_opciones,
                                   preseleccionada=viviendas[0] if len(viviendas) == 1 else None)
        try:
            viv_oid = oid_from_safe(vivienda_safe)
        except Exception:
            flash_error('Vivienda no válida.')
            return render_template('incidencias/form.html', form=form,
                                   viviendas_opciones=viviendas_opciones,
                                   preseleccionada=viviendas[0] if len(viviendas) == 1 else None)

        viv = db.session.get(Vivienda, viv_oid)
        if not viv:
            abort(404)

        inc = Incidencia(
            vivienda_oid=viv_oid,
            comunidad_oid=viv.comunidad_oid,
            usuario_oid=current_user.id,
            titulo=form.titulo.data.strip(),
            descripcion=form.descripcion.data.strip(),
            tipo=form.tipo.data,
        )
        db.session.add(inc)
        db.session.commit()
        flash_exito('Incidencia enviada correctamente.')
        return redirect(url_for('incidencias.lista'))

    preseleccionada = viviendas[0] if len(viviendas) == 1 else None
    return render_template('incidencias/form.html', form=form,
                           viviendas_opciones=viviendas_opciones,
                           preseleccionada=preseleccionada)


# ---------------------------------------------------------------------------
# Detalle de incidencia
# ---------------------------------------------------------------------------

@incidencias_bp.route('/<safe_oid>')
@login_required
def detalle(safe_oid):
    try:
        oid = oid_from_safe(safe_oid)
    except Exception:
        abort(404)
    inc = db.session.get(Incidencia, oid)
    if not inc:
        abort(404)

    # Permiso: superadmin ve todo; usuario solo las suyas
    if not current_user.es_superadmin:
        if inc.usuario_oid != current_user.id:
            abort(403)

    viv = db.session.get(Vivienda, inc.vivienda_oid)

    form_respuesta = RespuestaForm(prefix='resp')
    if inc.estado:
        form_respuesta.estado.data = inc.estado

    return render_template('incidencias/detalle.html',
                           inc=inc,
                           safe_oid=safe_oid,
                           viv=viv,
                           form_respuesta=form_respuesta)


# ---------------------------------------------------------------------------
# Responder incidencia (solo admin/superadmin)
# ---------------------------------------------------------------------------

def _puede_gestionar(inc) -> bool:
    """True si el usuario actual puede responder/gestionar esta incidencia."""
    return current_user.es_superadmin


@incidencias_bp.route('/<safe_oid>/responder', methods=['POST'])
@login_required
def responder(safe_oid):
    try:
        oid = oid_from_safe(safe_oid)
    except Exception:
        abort(404)
    inc = db.session.get(Incidencia, oid)
    if not inc:
        abort(404)
    if not _puede_gestionar(inc):
        abort(403)

    form_respuesta = RespuestaForm(prefix='resp')
    if form_respuesta.validate_on_submit():
        inc.respuesta_admin = form_respuesta.respuesta.data.strip()
        inc.estado = form_respuesta.estado.data
        if inc.estado == 'cerrada' and not inc.fecha_cierre:
            inc.fecha_cierre = date.today().isoformat()
        elif inc.estado != 'cerrada':
            inc.fecha_cierre = None
        db.session.commit()
        flash_exito('Respuesta guardada correctamente.')
    else:
        flash_error('Revisa los campos del formulario de respuesta.')

    return redirect(url_for('incidencias.detalle', safe_oid=safe_oid))


# ---------------------------------------------------------------------------
# Eliminar incidencia (solo superadmin)
# ---------------------------------------------------------------------------

@incidencias_bp.route('/<safe_oid>/eliminar', methods=['POST'])
@login_required
@superadmin_required
def eliminar(safe_oid):
    try:
        oid = oid_from_safe(safe_oid)
    except Exception:
        if is_xhr():
            return jsonify({'success': False, 'error': 'OID inválida'}), 404
        abort(404)

    inc = db.session.get(Incidencia, oid)
    if not inc:
        if is_xhr():
            return jsonify({'success': False, 'error': 'No encontrado'}), 404
        abort(404)

    db.session.delete(inc)
    db.session.commit()
    flash_exito('Incidencia eliminada.')

    if is_xhr():
        return jsonify({'success': True})
    return redirect(url_for('incidencias.lista'))
