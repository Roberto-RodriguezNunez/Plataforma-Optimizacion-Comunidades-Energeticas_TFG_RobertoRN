"""Formularios de CierreMensual."""
import re
from flask_wtf import FlaskForm
from wtforms import FloatField, StringField, SelectField, SubmitField
from wtforms.validators import DataRequired, InputRequired, NumberRange, Regexp


class CierreForm(FlaskForm):
    mes = StringField('Mes (YYYY-MM)', validators=[
        DataRequired(),
        Regexp(r'^\d{4}-\d{2}$', message='Formato requerido: YYYY-MM')
    ])
    consumo_total_kwh = FloatField('Consumo total (kWh)',
                                   validators=[InputRequired(), NumberRange(min=0)])
    autoconsumo_directo_kwh = FloatField('Autoconsumo directo (kWh)',
                                         validators=[InputRequired(), NumberRange(min=0)])
    energia_de_bateria_kwh = FloatField('Energía de batería (kWh)',
                                        validators=[InputRequired(), NumberRange(min=0)])
    vertido_a_red_kwh = FloatField('Vertido a red (kWh)',
                                   validators=[InputRequired(), NumberRange(min=0)])
    ahorro_eur = FloatField('Ahorro (€)', validators=[InputRequired(), NumberRange(min=0)])
    factura_escenario_base_eur = FloatField('Factura sin comunidad (€)',
                                            validators=[InputRequired(), NumberRange(min=0)])
    factura_escenario_real_eur = FloatField('Factura con comunidad (€)',
                                            validators=[InputRequired(), NumberRange(min=0)])
    porcentaje_ahorro_global = FloatField('% del ahorro global',
                                          validators=[InputRequired(), NumberRange(min=0, max=100)])
    coeficiente_reparto_aplicado = FloatField('Coeficiente de reparto aplicado',
                                              validators=[InputRequired(), NumberRange(min=0, max=1)])
    enviar = SubmitField('Guardar cierre')
