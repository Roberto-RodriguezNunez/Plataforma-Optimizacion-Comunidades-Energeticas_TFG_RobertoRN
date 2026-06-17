"""Formularios de AccesoVivienda."""
from flask_wtf import FlaskForm
from wtforms import SelectField, DateField, SubmitField
from wtforms.validators import DataRequired, Optional


class AccesoForm(FlaskForm):
    usuario_safe_oid = SelectField('Usuario', validators=[DataRequired()])
    rol_en_vivienda = SelectField('Rol en la vivienda', validators=[DataRequired()],
                                   choices=[('titular', 'Titular'),
                                            ('convivente', 'Convivente'),
                                            ('solo_lectura', 'Solo lectura')])
    fecha_incorporacion = DateField('Fecha de incorporación', validators=[Optional()])
    enviar = SubmitField('Guardar')
