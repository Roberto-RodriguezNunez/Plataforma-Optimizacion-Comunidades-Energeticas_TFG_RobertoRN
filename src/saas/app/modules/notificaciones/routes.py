"""Rutas de Notificaciones."""
from flask import render_template, redirect, url_for, request, abort, jsonify
from flask_login import login_required, current_user

from app.extensions import db
from app.modules.notificaciones import notificaciones_bp
from app.modules.notificaciones.forms import NotificacionForm
from app.models.notificacion import Notificacion
from app.models.usuario import Usuario
from app.helpers import oid_from_safe, oid_to_safe, flash_exito, flash_error, is_xhr, crear_notificacion
from app.decorators import superadmin_required


@notificaciones_bp.route('/')
@login_required
def lista():
    notificaciones = Notificacion.query.filter_by(
        usuario_oid=current_user.id
    ).order_by(Notificacion.fecha.desc()).all()
    return render_template('notificaciones/lista.html', notificaciones=notificaciones)


@notificaciones_bp.route('/<safe_oid>/leer', methods=['POST'])
@login_required
def marcar_leida(safe_oid):
    try:
        oid = oid_from_safe(safe_oid)
    except Exception:
        if is_xhr():
            return jsonify({'success': False, 'error': 'OID inválida'}), 404
        abort(404)
    notif = db.session.get(Notificacion, oid)
    if not notif:
        if is_xhr():
            return jsonify({'success': False, 'error': 'No encontrada'}), 404
        abort(404)
    if notif.usuario_oid != current_user.id:
        abort(403)

    notif.marcar_leida()
    db.session.commit()

    if is_xhr():
        return jsonify({'success': True})
    return redirect(url_for('notificaciones.lista'))


@notificaciones_bp.route('/leer-todas', methods=['POST'])
@login_required
def marcar_todas_leidas():
    Notificacion.query.filter_by(
        usuario_oid=current_user.id, leida=False
    ).update({'leida': True})
    db.session.commit()
    flash_exito('Todas las notificaciones marcadas como leídas.')
    if is_xhr():
        return jsonify({'success': True})
    return redirect(url_for('notificaciones.lista'))


@notificaciones_bp.route('/<safe_oid>/eliminar', methods=['POST'])
@login_required
def eliminar(safe_oid):
    try:
        oid = oid_from_safe(safe_oid)
    except Exception:
        if is_xhr():
            return jsonify({'success': False, 'error': 'OID inválida'}), 404
        abort(404)
    notif = db.session.get(Notificacion, oid)
    if not notif:
        if is_xhr():
            return jsonify({'success': False, 'error': 'No encontrada'}), 404
        abort(404)
    if notif.usuario_oid != current_user.id and not current_user.es_superadmin:
        abort(403)

    db.session.delete(notif)
    db.session.commit()
    flash_exito('Notificación eliminada.')
    if is_xhr():
        return jsonify({'success': True})
    return redirect(url_for('notificaciones.lista'))


@notificaciones_bp.route('/nueva', methods=['GET', 'POST'])
@login_required
@superadmin_required
def nueva():
    form = NotificacionForm()

    # Construir opciones de destinatario
    usuarios = Usuario.query.all()
    opciones = [('__todos__', 'Todos los usuarios')]
    for u in sorted(usuarios, key=lambda u: u.nombre):
        opciones.append((oid_to_safe(u.__oid__), f'{u.nombre} ({u.email})'))
    form.destinatario.choices = opciones

    if form.validate_on_submit():
        titulo = form.titulo.data.strip()
        mensaje = form.mensaje.data.strip()
        dest = form.destinatario.data

        if dest == '__todos__':
            for u in usuarios:
                crear_notificacion(u.id, 'general', titulo, mensaje)
            flash_exito(f'Notificación enviada a {len(usuarios)} usuarios.')
        else:
            try:
                usr_oid = oid_from_safe(dest)
            except Exception:
                flash_error('Usuario no válido.')
                return render_template('notificaciones/form.html', form=form)
            crear_notificacion(usr_oid, 'general', titulo, mensaje)
            usr = db.session.get(Usuario, usr_oid)
            flash_exito(f'Notificación enviada a {usr.nombre if usr else "usuario"}.')

        return redirect(url_for('notificaciones.lista'))

    return render_template('notificaciones/form.html', form=form)
