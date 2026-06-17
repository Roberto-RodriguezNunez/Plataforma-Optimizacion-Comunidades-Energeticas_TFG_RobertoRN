"""Formularios de Vivienda."""
from flask_wtf import FlaskForm
from wtforms import (StringField, FloatField, BooleanField, SelectField,
                     DateField, IntegerField, SubmitField)
from wtforms.validators import DataRequired, Optional, NumberRange, Length


class ViviendaForm(FlaskForm):
    identificador = StringField('Identificador', validators=[DataRequired(), Length(1, 80)])
    direccion_completa = StringField('Dirección completa', validators=[Optional(), Length(max=200)])
    cups = StringField('CUPS', validators=[Optional(), Length(max=30)])
    potencia_contratada_kw = FloatField('Potencia contratada (kW)',
                                        validators=[DataRequired(), NumberRange(min=0)])
    fecha_alta = DateField('Fecha de alta', validators=[DataRequired()])
    tiene_paneles = BooleanField('Tiene paneles solares')
    potencia_pico_paneles_kwp = FloatField('Potencia pico paneles (kWp)',
                                           validators=[Optional(), NumberRange(min=0)])
    numero_paneles = IntegerField('Número de paneles', validators=[Optional(), NumberRange(min=1)])
    fecha_instalacion_paneles = DateField('Fecha instalación paneles', validators=[Optional()])
    orientacion_paneles = SelectField('Orientación paneles', validators=[Optional()],
                                      choices=[('', 'Sin especificar'), ('sur', 'Sur'),
                                               ('este', 'Este'), ('oeste', 'Oeste'),
                                               ('norte', 'Norte'), ('mixta', 'Mixta')])
    enviar = SubmitField('Guardar')
