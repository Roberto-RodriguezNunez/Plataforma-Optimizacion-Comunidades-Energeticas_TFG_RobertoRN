"""Modelo AccesoVivienda: relación N:M entre Usuario y Vivienda."""
from datetime import date

from app.extensions import db


class AccesoVivienda(db.Model):
    """Relaciona un usuario con una vivienda y define su rol en ella.

    Un usuario puede tener acceso a varias viviendas (incluso de comunidades
    distintas) y una vivienda puede tener varios usuarios con acceso.
    """

    __tablename__ = 'accesos'

    ROLES = ('titular', 'convivente', 'solo_lectura')

    id = db.Column(db.Integer, primary_key=True)
    usuario_oid = db.Column(db.Integer, db.ForeignKey('usuarios.id'),
                            nullable=False, index=True)
    vivienda_oid = db.Column(db.Integer, db.ForeignKey('viviendas.id'),
                             nullable=False, index=True)
    rol_en_vivienda = db.Column(db.String(20), nullable=False, default='titular')
    fecha_incorporacion = db.Column(db.String(40))

    def __init__(self, usuario_oid, vivienda_oid, rol_en_vivienda='titular',
                 fecha_incorporacion=None):
        self.usuario_oid = int(usuario_oid)
        self.vivienda_oid = int(vivienda_oid)
        self.rol_en_vivienda = rol_en_vivienda
        self.fecha_incorporacion = fecha_incorporacion or date.today().isoformat()

    @property
    def __oid__(self):
        return self.id

    def __repr__(self):
        return f'<AccesoVivienda u={self.usuario_oid} v={self.vivienda_oid} rol={self.rol_en_vivienda}>'
