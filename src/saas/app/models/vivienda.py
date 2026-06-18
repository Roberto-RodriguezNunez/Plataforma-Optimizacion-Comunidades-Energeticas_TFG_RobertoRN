"""Modelo de Vivienda dentro de una comunidad energética."""
from datetime import date

from app.extensions import db


class Vivienda(db.Model):
    """Representa una vivienda asociada a una comunidad energética.

    Almacena datos técnicos del punto de suministro y, opcionalmente,
    la instalación fotovoltaica propia.
    """

    __tablename__ = 'viviendas'

    ORIENTACIONES = ('sur', 'este', 'oeste', 'norte', 'mixta')

    id = db.Column(db.Integer, primary_key=True)
    comunidad_oid = db.Column(db.Integer, db.ForeignKey('comunidades.id'),
                              nullable=False, index=True)
    identificador = db.Column(db.String(120), nullable=False)
    direccion_completa = db.Column(db.String(255), default='')
    cups = db.Column(db.String(40), default='')
    potencia_contratada_kw = db.Column(db.Float, default=0.0)
    coeficiente_reparto = db.Column(db.Float, default=0.0)
    fecha_alta = db.Column(db.String(40))
    tiene_paneles = db.Column(db.Boolean, default=False)
    potencia_pico_paneles_kwp = db.Column(db.Float, nullable=True)
    numero_paneles = db.Column(db.Integer, nullable=True)
    fecha_instalacion_paneles = db.Column(db.String(40), nullable=True)
    orientacion_paneles = db.Column(db.String(20), nullable=True)

    def __init__(self, comunidad_oid, identificador, direccion_completa='',
                 cups='', potencia_contratada_kw=0.0, coeficiente_reparto=0.0,
                 fecha_alta=None, tiene_paneles=False,
                 potencia_pico_paneles_kwp=None, numero_paneles=None,
                 fecha_instalacion_paneles=None, orientacion_paneles=None):
        self.comunidad_oid = int(comunidad_oid)
        self.identificador = identificador
        self.direccion_completa = direccion_completa
        self.cups = cups
        self.potencia_contratada_kw = float(potencia_contratada_kw)
        self.coeficiente_reparto = float(coeficiente_reparto)
        self.fecha_alta = fecha_alta or date.today().isoformat()
        self.tiene_paneles = bool(tiene_paneles)
        # Campos de paneles: solo relevantes si tiene_paneles=True
        self.potencia_pico_paneles_kwp = (
            float(potencia_pico_paneles_kwp) if potencia_pico_paneles_kwp is not None else None
        )
        self.numero_paneles = int(numero_paneles) if numero_paneles is not None else None
        self.fecha_instalacion_paneles = fecha_instalacion_paneles
        self.orientacion_paneles = orientacion_paneles

    @property
    def __oid__(self):
        return self.id

    def __repr__(self):
        return f'<Vivienda {self.identificador}>'
