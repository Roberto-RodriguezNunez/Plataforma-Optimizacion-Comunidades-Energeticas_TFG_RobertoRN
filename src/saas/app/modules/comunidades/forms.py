"""Formularios de Comunidad."""
from flask_wtf import FlaskForm
from wtforms import StringField, TextAreaField, SelectField, DateField, SubmitField
from wtforms.validators import DataRequired, Length, Optional


class ComunidadForm(FlaskForm):
    nombre = StringField('Nombre', validators=[DataRequired(), Length(2, 120)])
    ubicacion = StringField('Ubicación', validators=[DataRequired(), Length(2, 120)])
    fecha_constitucion = DateField('Fecha de constitución', validators=[DataRequired()])
    estado = SelectField('Estado', choices=[('activa', 'Activa'), ('suspendida', 'Suspendida')])
    descripcion = TextAreaField('Descripción', validators=[Optional(), Length(max=500)])
    enviar = SubmitField('Guardar')
