"""Formularios del módulo de Incidencias."""
from flask_wtf import FlaskForm
from wtforms import StringField, TextAreaField, SelectField, SubmitField
from wtforms.validators import DataRequired, Length


class IncidenciaForm(FlaskForm):
    titulo = StringField('Título', validators=[
        DataRequired(message='El título es obligatorio.'),
        Length(max=200, message='Máximo 200 caracteres.')
    ])
    descripcion = TextAreaField('Descripción', validators=[
        DataRequired(message='La descripción es obligatoria.')
    ])
    tipo = SelectField('Tipo', validators=[DataRequired()],
                       choices=[
                           ('averia', 'Avería'),
                           ('consulta', 'Consulta'),
                           ('otro', 'Otro'),
                       ])
    enviar = SubmitField('Enviar incidencia')


class RespuestaForm(FlaskForm):
    respuesta = TextAreaField('Respuesta', validators=[
        DataRequired(message='La respuesta no puede estar vacía.')
    ])
    estado = SelectField('Estado', validators=[DataRequired()],
                         choices=[
                             ('abierta', 'Abierta'),
                             ('en_proceso', 'En proceso'),
                             ('cerrada', 'Cerrada'),
                         ])
    enviar = SubmitField('Guardar respuesta')
