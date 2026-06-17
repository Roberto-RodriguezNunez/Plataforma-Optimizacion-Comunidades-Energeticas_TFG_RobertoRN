"""Formularios de Notificaciones."""
from flask_wtf import FlaskForm
from wtforms import StringField, TextAreaField, SelectField, SubmitField
from wtforms.validators import DataRequired, Length


class NotificacionForm(FlaskForm):
    destinatario = SelectField('Enviar a', validators=[DataRequired()])
    titulo = StringField('Título', validators=[DataRequired(), Length(2, 120)])
    mensaje = TextAreaField('Mensaje', validators=[DataRequired(), Length(2, 1000)])
    enviar = SubmitField('Enviar notificación')
