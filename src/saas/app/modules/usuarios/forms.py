"""Formularios de Usuario."""
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired, Length


class PerfilForm(FlaskForm):
    nombre = StringField('Nombre', validators=[DataRequired(), Length(2, 100)])
    enviar = SubmitField('Guardar cambios')
