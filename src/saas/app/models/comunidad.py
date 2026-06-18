"""Modelo de Comunidad energética."""
from datetime import date

from app.extensions import db


class Comunidad(db.Model):
    """Representa una comunidad energética en la plataforma.

    Una comunidad agrupa viviendas que comparten infraestructura energética
    (batería, generación solar) y reparten ahorros según coeficientes.
    """

    __tablename__ = 'comunidades'

    ESTADOS = ('activa', 'suspendida')

    id = db.Column(db.Integer, primary_key=True)
    nombre = db.Column(db.String(120), nullable=False)
    ubicacion = db.Column(db.String(200), nullable=False)
    fecha_constitucion = db.Column(db.String(40))
    estado = db.Column(db.String(20), nullable=False, default='activa')
    descripcion = db.Column(db.Text, default='')

    def __init__(self, nombre, ubicacion, fecha_constitucion=None,
                 estado='activa', descripcion=''):
        self.nombre = nombre
        self.ubicacion = ubicacion
        self.fecha_constitucion = fecha_constitucion or date.today().isoformat()
        self.estado = estado
        self.descripcion = descripcion

    @property
    def __oid__(self):
        return self.id

    @property
    def es_activa(self):
        return self.estado == 'activa'

    def __repr__(self):
        return f'<Comunidad {self.nombre}>'
