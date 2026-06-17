"""Gestión de AccesoVivienda."""
from flask import render_template, redirect, url_for, request, abort, jsonify
from flask_login import login_required, current_user
from datetime import date

from app.extensions import db
from app.modules.accesos import accesos_bp
from app.modules.accesos.forms import AccesoForm
from app.models.acceso import AccesoVivienda
from app.models.vivienda import Vivienda
from app.models.usuario import Usuario
from app.models.comunidad import Comunidad
from app.helpers import oid_from_safe, oid_to_safe, flash_exito, flash_error, is_xhr
from app.decorators import superadmin_required


@accesos_bp.route('/vivienda/<vivienda_safe_oid>/')
@login_required
@superadmin_required
def lista(vivienda_safe_oid):
    try:
        viv_oid = oid_from_safe(vivienda_safe_oid)
    except Exception:
        abort(404)
    viv = db.session.get(Vivienda, viv_oid)
    if not viv:
        abort(404)

    accesos = AccesoVivienda.query.filter_by(vivienda_oid=viv_oid).all()
    accesos_detalle = []
    for a in accesos:
        usr = db.session.get(Usuario, a.usuario_oid)
        accesos_detalle.append({'acceso': a, 'usuario': usr,
                                'safe_oid': oid_to_safe(a.__oid__)})

    com = db.session.get(Comunidad, viv.comunidad_oid)
    return render_template('accesos/lista.html',
                           accesos_detalle=accesos_detalle,
                           viv=viv, vivienda_safe_oid=vivienda_safe_oid,
                           com=com)


@accesos_bp.route('/vivienda/<vivienda_safe_oid>/nuevo', methods=['GET', 'POST'])
@login_required
@superadmin_required
def nuevo(vivienda_safe_oid):
    try:
        viv_oid = oid_from_safe(vivienda_safe_oid)
    except Exception:
        abort(404)
    viv = db.session.get(Vivienda, viv_oid)
    if not viv:
        abort(404)

    form = AccesoForm()
    todos_usuarios = Usuario.query.all()
    form.usuario_safe_oid.choices = [
        (oid_to_safe(u.__oid__), f'{u.nombre} ({u.email})')
        for u in todos_usuarios
    ]

    if form.validate_on_submit():
        try:
            usr_oid = oid_from_safe(form.usuario_safe_oid.data)
        except Exception:
            flash_error('Usuario no válido.')
            return render_template('accesos/form.html', form=form, viv=viv,
                                   vivienda_safe_oid=vivienda_safe_oid)

        existe = AccesoVivienda.query.filter_by(
            usuario_oid=usr_oid, vivienda_oid=viv_oid
        ).first()
        if existe:
            flash_error('Este usuario ya tiene acceso a esta vivienda.')
            return render_template('accesos/form.html', form=form, viv=viv,
                                   vivienda_safe_oid=vivienda_safe_oid)

        fecha = form.fecha_incorporacion.data.isoformat() if form.fecha_incorporacion.data else date.today().isoformat()
        acceso = AccesoVivienda(
            usuario_oid=usr_oid,
            vivienda_oid=viv_oid,
            rol_en_vivienda=form.rol_en_vivienda.data,
            fecha_incorporacion=fecha
        )
        db.session.add(acceso)
        db.session.commit()
        flash_exito('Acceso creado correctamente.')
        return redirect(url_for('accesos.lista', vivienda_safe_oid=vivienda_safe_oid))

    return render_template('accesos/form.html', form=form, viv=viv,
                           vivienda_safe_oid=vivienda_safe_oid, titulo='Nuevo acceso')


@accesos_bp.route('/<safe_oid>/editar', methods=['GET', 'POST'])
@login_required
@superadmin_required
def editar(safe_oid):
    try:
        oid = oid_from_safe(safe_oid)
    except Exception:
        abort(404)
    acceso = db.session.get(AccesoVivienda, oid)
    if not acceso:
        abort(404)

    viv = db.session.get(Vivienda, acceso.vivienda_oid)
    if not viv:
        abort(404)

    vivienda_safe_oid = oid_to_safe(acceso.vivienda_oid)
    form = AccesoForm(obj=acceso)
    todos_usuarios = Usuario.query.all()
    form.usuario_safe_oid.choices = [
        (oid_to_safe(u.__oid__), f'{u.nombre} ({u.email})')
        for u in todos_usuarios
    ]
    if request.method == 'GET':
        form.usuario_safe_oid.data = oid_to_safe(acceso.usuario_oid)
        if acceso.fecha_incorporacion:
            try:
                form.fecha_incorporacion.data = date.fromisoformat(acceso.fecha_incorporacion)
            except Exception:
                pass

    if form.validate_on_submit():
        acceso.rol_en_vivienda = form.rol_en_vivienda.data
        if form.fecha_incorporacion.data:
            acceso.fecha_incorporacion = form.fecha_incorporacion.data.isoformat()
        db.session.commit()
        flash_exito('Acceso actualizado.')
        return redirect(url_for('accesos.lista', vivienda_safe_oid=vivienda_safe_oid))

    return render_template('accesos/form.html', form=form, viv=viv,
                           vivienda_safe_oid=vivienda_safe_oid, titulo='Editar acceso')


@accesos_bp.route('/<safe_oid>/eliminar', methods=['POST'])
@login_required
def eliminar(safe_oid):
    try:
        oid = oid_from_safe(safe_oid)
    except Exception:
        if is_xhr():
            return jsonify({'success': False, 'error': 'OID inválida'}), 404
        abort(404)
    acceso = db.session.get(AccesoVivienda, oid)
    if not acceso:
        if is_xhr():
            return jsonify({'success': False, 'error': 'No encontrado'}), 404
        abort(404)

    # Solo superadmin puede eliminar desde la lista; la ruta de usuario detalle también llega aquí
    if not current_user.es_superadmin:
        abort(403)

    vivienda_safe_oid = oid_to_safe(acceso.vivienda_oid)

    # Guardia: no borrar el único titular
    if acceso.rol_en_vivienda == 'titular':
        otros_titulares = AccesoVivienda.query.filter(
            AccesoVivienda.vivienda_oid == acceso.vivienda_oid,
            AccesoVivienda.rol_en_vivienda == 'titular',
            AccesoVivienda.id != oid,
        ).first()
        if not otros_titulares:
            msg = 'No se puede eliminar el único titular de la vivienda.'
            flash_error(msg)
            if is_xhr():
                return jsonify({'success': False, 'error': msg})
            return redirect(url_for('accesos.lista', vivienda_safe_oid=vivienda_safe_oid))

    db.session.delete(acceso)
    db.session.commit()
    flash_exito('Acceso eliminado.')
    if is_xhr():
        return jsonify({'success': True})
    return redirect(url_for('accesos.lista', vivienda_safe_oid=vivienda_safe_oid))
