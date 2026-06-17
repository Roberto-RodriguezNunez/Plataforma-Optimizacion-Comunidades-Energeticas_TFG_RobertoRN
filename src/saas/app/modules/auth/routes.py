"""Rutas de autenticación: login, logout, registro, perfil."""
from flask import render_template, redirect, url_for, request
from flask_login import login_user, logout_user, login_required, current_user

from app.extensions import db
from app.modules.auth import auth_bp
from app.modules.auth.forms import LoginForm, RegistroForm, CambiarPasswordForm
from app.helpers import flash_exito, flash_error
from app.models.usuario import Usuario


@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('main.index'))
    form = LoginForm()
    if form.validate_on_submit():
        email_buscado = form.email.data.lower().strip()
        usuario = Usuario.query.filter_by(email=email_buscado).first()
        if usuario and usuario.verificar_password(form.password.data):
            login_user(usuario, remember=form.recordarme.data)
            siguiente = request.args.get('next')
            return redirect(siguiente or url_for('main.index'))
        flash_error('Email o contraseña incorrectos.')
    return render_template('auth/login.html', form=form)


@auth_bp.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('auth.login'))


@auth_bp.route('/registro', methods=['GET', 'POST'])
def registro():
    if current_user.is_authenticated:
        return redirect(url_for('main.index'))
    form = RegistroForm()
    if form.validate_on_submit():
        email = form.email.data.lower().strip()
        if Usuario.query.filter_by(email=email).first():
            flash_error('Este email ya está registrado.')
            return render_template('auth/registro.html', form=form)
        # El primer usuario registrado es automáticamente superadmin
        es_primero = Usuario.query.count() == 0
        rol = 'superadmin' if es_primero else 'normal'
        usuario = Usuario(form.nombre.data, email, form.password.data, rol)
        db.session.add(usuario)
        db.session.commit()
        login_user(usuario)
        flash_exito('¡Cuenta creada correctamente! Bienvenido a LeaLink.')
        return redirect(url_for('main.index'))
    return render_template('auth/registro.html', form=form)


@auth_bp.route('/perfil', methods=['GET', 'POST'])
@login_required
def perfil():
    from app.modules.usuarios.forms import PerfilForm
    form = PerfilForm(obj=current_user)
    if form.validate_on_submit():
        current_user.nombre = form.nombre.data
        db.session.commit()
        flash_exito('Perfil actualizado.')
        return redirect(url_for('auth.perfil'))
    return render_template('auth/perfil.html', form=form)


@auth_bp.route('/cambiar-password', methods=['GET', 'POST'])
@login_required
def cambiar_password():
    form = CambiarPasswordForm()
    if form.validate_on_submit():
        if not current_user.verificar_password(form.password_actual.data):
            flash_error('La contraseña actual no es correcta.')
            return render_template('auth/cambiar_password.html', form=form)
        current_user.set_password(form.nueva_password.data)
        db.session.commit()
        flash_exito('Contraseña actualizada correctamente.')
        return redirect(url_for('auth.perfil'))
    return render_template('auth/cambiar_password.html', form=form)
