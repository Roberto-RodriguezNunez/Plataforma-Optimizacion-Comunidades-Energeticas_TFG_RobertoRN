"""Formularios de autenticación."""
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SubmitField
from wtforms.validators import DataRequired, Email, EqualTo, Length


class LoginForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Contraseña', validators=[DataRequired()])
    recordarme = BooleanField('Recordarme')
    enviar = SubmitField('Iniciar sesión')


class RegistroForm(FlaskForm):
    nombre = StringField('Nombre', validators=[DataRequired(), Length(2, 100)])
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Contraseña', validators=[DataRequired(), Length(min=8, message='Mínimo 8 caracteres')])
    password2 = PasswordField('Repetir contraseña', validators=[
        DataRequired(), EqualTo('password', message='Las contraseñas no coinciden')
    ])
    enviar = SubmitField('Registrarse')


class CambiarPasswordForm(FlaskForm):
    password_actual = PasswordField('Contraseña actual', validators=[DataRequired()])
    nueva_password = PasswordField('Nueva contraseña', validators=[DataRequired(), Length(min=8)])
    nueva_password2 = PasswordField('Repetir nueva contraseña', validators=[
        DataRequired(), EqualTo('nueva_password', message='Las contraseñas no coinciden')
    ])
    enviar = SubmitField('Cambiar contraseña')
