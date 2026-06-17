"""CRUD de Baterías."""
from flask import render_template, redirect, url_for, request, abort, jsonify
from flask_login import login_required, current_user
from datetime import date

from app.extensions import db
from app.modules.baterias import baterias_bp
from app.modules.baterias.forms import BateriaForm, CambiarEstadoBateriaForm
from app.models.bateria import Bateria
from app.models.comunidad import Comunidad
from app.helpers import (oid_from_safe, oid_to_safe, flash_exito, flash_error,
                          is_xhr, notificar_a_comunidad)
from app.decorators import superadmin_required


def _comprobar_superadmin():
    if not current_user.es_superadmin:
        abort(403)


@baterias_bp.route('/comunidad/<comunidad_safe_oid>/')
@login_required
def detalle_comunidad(comunidad_safe_oid):
    try:
        com_oid = oid_from_safe(comunidad_safe_oid)
    except Exception:
        abort(404)
    com = db.session.get(Comunidad, com_oid)
    if not com:
        abort(404)
    _comprobar_superadmin()

    bateria = Bateria.query.filter_by(comunidad_oid=com_oid).first()
    form_estado = CambiarEstadoBateriaForm()
    if bateria:
        form_estado.estado.data = bateria.estado

    return render_template('baterias/detalle.html',
                           bateria=bateria, com=com,
                           comunidad_safe_oid=comunidad_safe_oid,
                           form_estado=form_estado,
                           safe_oid=oid_to_safe(bateria.__oid__) if bateria else None)


@baterias_bp.route('/comunidad/<comunidad_safe_oid>/nueva', methods=['GET', 'POST'])
@login_required
def nueva(comunidad_safe_oid):
    try:
        com_oid = oid_from_safe(comunidad_safe_oid)
    except Exception:
        abort(404)
    com = db.session.get(Comunidad, com_oid)
    if not com:
        abort(404)
    _comprobar_superadmin()

    # Solo puede haber una batería por comunidad
    if Bateria.query.filter_by(comunidad_oid=com_oid).first():
        flash_error('Esta comunidad ya tiene una batería registrada. Edítala en lugar de crear una nueva.')
        return redirect(url_for('baterias.detalle_comunidad', comunidad_safe_oid=comunidad_safe_oid))

    form = BateriaForm()
    if form.validate_on_submit():
        bat = Bateria(
            comunidad_oid=com_oid,
            capacidad_nominal_kwh=form.capacidad_nominal_kwh.data,
            capacidad_util_actual_kwh=form.capacidad_util_actual_kwh.data,
            ciclos_acumulados=form.ciclos_acumulados.data,
            fecha_instalacion=form.fecha_instalacion.data.isoformat(),
            fabricante=form.fabricante.data or '',
            modelo=form.modelo.data or '',
            estado=form.estado.data
        )
        db.session.add(bat)
        db.session.commit()
        flash_exito('Batería registrada correctamente.')
        return redirect(url_for('baterias.detalle_comunidad', comunidad_safe_oid=comunidad_safe_oid))
    return render_template('baterias/form.html', form=form,
                           titulo='Nueva batería', com=com,
                           comunidad_safe_oid=comunidad_safe_oid)


@baterias_bp.route('/<safe_oid>/editar', methods=['GET', 'POST'])
@login_required
def editar(safe_oid):
    try:
        oid = oid_from_safe(safe_oid)
    except Exception:
        abort(404)
    bat = db.session.get(Bateria, oid)
    if not bat:
        abort(404)
    com_oid = bat.comunidad_oid
    _comprobar_superadmin()
    com = db.session.get(Comunidad, com_oid)
    comunidad_safe_oid = oid_to_safe(com_oid)

    form = BateriaForm(obj=bat)
    if request.method == 'GET':
        try:
            form.fecha_instalacion.data = date.fromisoformat(bat.fecha_instalacion)
        except Exception:
            pass

    if form.validate_on_submit():
        bat.capacidad_nominal_kwh = form.capacidad_nominal_kwh.data
        bat.capacidad_util_actual_kwh = form.capacidad_util_actual_kwh.data
        bat.ciclos_acumulados = form.ciclos_acumulados.data
        bat.fecha_instalacion = form.fecha_instalacion.data.isoformat()
        bat.fabricante = form.fabricante.data or ''
        bat.modelo = form.modelo.data or ''
        bat.estado = form.estado.data
        db.session.commit()
        flash_exito('Batería actualizada.')
        return redirect(url_for('baterias.detalle_comunidad', comunidad_safe_oid=comunidad_safe_oid))
    return render_template('baterias/form.html', form=form,
                           titulo='Editar batería', com=com,
                           comunidad_safe_oid=comunidad_safe_oid, safe_oid=safe_oid)


@baterias_bp.route('/<safe_oid>/cambiar-estado', methods=['POST'])
@login_required
def cambiar_estado(safe_oid):
    """Cambia el estado de la batería. Notifica a la comunidad si pasa a mantenimiento."""
    try:
        oid = oid_from_safe(safe_oid)
    except Exception:
        if is_xhr():
            return jsonify({'success': False, 'error': 'OID inválida'}), 404
        abort(404)
    bat = db.session.get(Bateria, oid)
    if not bat:
        if is_xhr():
            return jsonify({'success': False, 'error': 'No encontrada'}), 404
        abort(404)

    com_oid = bat.comunidad_oid
    _comprobar_superadmin()

    nuevo_estado = request.form.get('estado', bat.estado)
    if nuevo_estado not in Bateria.ESTADOS:
        flash_error('Estado no válido.')
        if is_xhr():
            return jsonify({'success': False, 'error': 'Estado no válido'})
        return redirect(url_for('baterias.detalle_comunidad',
                                comunidad_safe_oid=oid_to_safe(com_oid)))

    estado_anterior = bat.estado
    bat.estado = nuevo_estado
    db.session.commit()

    # Notificar si pasa a mantenimiento o avería
    if nuevo_estado in ('mantenimiento', 'averiada') and estado_anterior != nuevo_estado:
        notificar_a_comunidad(
            com_oid,
            tipo='bateria_mantenimiento',
            titulo=f'Batería en {nuevo_estado}',
            mensaje=f'La batería de la comunidad ha cambiado a estado: {nuevo_estado}.',
            entidad_oid=oid,
            entidad_tipo='Bateria'
        )

    flash_exito(f'Estado de la batería cambiado a: {nuevo_estado}.')
    comunidad_safe_oid = oid_to_safe(com_oid)
    if is_xhr():
        return jsonify({'success': True, 'nuevo_estado': nuevo_estado,
                        'redirect': url_for('baterias.detalle_comunidad',
                                            comunidad_safe_oid=comunidad_safe_oid)})
    return redirect(url_for('baterias.detalle_comunidad', comunidad_safe_oid=comunidad_safe_oid))


@baterias_bp.route('/<safe_oid>/eliminar', methods=['POST'])
@login_required
def eliminar(safe_oid):
    try:
        oid = oid_from_safe(safe_oid)
    except Exception:
        if is_xhr():
            return jsonify({'success': False, 'error': 'OID inválida'}), 404
        abort(404)
    bat = db.session.get(Bateria, oid)
    if not bat:
        if is_xhr():
            return jsonify({'success': False, 'error': 'No encontrada'}), 404
        abort(404)

    com_oid = bat.comunidad_oid
    _comprobar_superadmin()
    comunidad_safe_oid = oid_to_safe(com_oid)

    db.session.delete(bat)
    db.session.commit()
    flash_exito('Batería eliminada.')
    if is_xhr():
        return jsonify({'success': True, 'redirect': url_for('comunidades.detalle', safe_oid=comunidad_safe_oid)})
    return redirect(url_for('comunidades.detalle', safe_oid=comunidad_safe_oid))
