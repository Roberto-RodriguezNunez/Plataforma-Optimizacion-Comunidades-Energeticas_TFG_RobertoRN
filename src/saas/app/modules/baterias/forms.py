"""Formularios de Batería."""
from flask_wtf import FlaskForm
from wtforms import FloatField, IntegerField, StringField, SelectField, DateField, SubmitField
from wtforms.validators import DataRequired, Optional, NumberRange, Length


class BateriaForm(FlaskForm):
    capacidad_nominal_kwh = FloatField('Capacidad nominal (kWh)',
                                       validators=[DataRequired(), NumberRange(min=0.1)])
    capacidad_util_actual_kwh = FloatField('Capacidad útil actual (kWh)',
                                           validators=[DataRequired(), NumberRange(min=0)])
    ciclos_acumulados = IntegerField('Ciclos acumulados',
                                     validators=[DataRequired(), NumberRange(min=0)])
    fecha_instalacion = DateField('Fecha de instalación', validators=[DataRequired()])
    fabricante = StringField('Fabricante', validators=[Optional(), Length(max=100)])
    modelo = StringField('Modelo', validators=[Optional(), Length(max=100)])
    estado = SelectField('Estado', validators=[DataRequired()],
                         choices=[('operativa', 'Operativa'),
                                  ('mantenimiento', 'En mantenimiento'),
                                  ('averiada', 'Averiada'),
                                  ('retirada', 'Retirada')])
    enviar = SubmitField('Guardar')


class CambiarEstadoBateriaForm(FlaskForm):
    estado = SelectField('Nuevo estado', validators=[DataRequired()],
                         choices=[('operativa', 'Operativa'),
                                  ('mantenimiento', 'En mantenimiento'),
                                  ('averiada', 'Averiada'),
                                  ('retirada', 'Retirada')])
    enviar = SubmitField('Cambiar estado')
