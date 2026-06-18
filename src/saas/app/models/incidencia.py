"""Modelo de Incidencia para usuarios de la plataforma."""
from datetime import date

from app.extensions import db


class Incidencia(db.Model):
    """Incidencia creada por un usuario sobre su vivienda.

    Puede ser una avería, consulta u otro tipo de comunicación.
    El administrador de la comunidad o el superadmin pueden responder y cerrarla.
    """

    __tablename__ = 'incidencias'

    TIPOS = ('averia', 'consulta', 'otro')
    ESTADOS = ('abierta', 'en_proceso', 'cerrada')

    id = db.Column(db.Integer, primary_key=True)
    vivienda_oid = db.Column(db.Integer, db.ForeignKey('viviendas.id'),
                             nullable=False, index=True)
    comunidad_oid = db.Column(db.Integer, db.ForeignKey('comunidades.id'),
                              nullable=False, index=True)
    usuario_oid = db.Column(db.Integer, db.ForeignKey('usuarios.id'),
                            nullable=False, index=True)
    titulo = db.Column(db.String(200), nullable=False)
    descripcion = db.Column(db.Text, nullable=False)
    tipo = db.Column(db.String(20), nullable=False, default='consulta')
    estado = db.Column(db.String(20), nullable=False, default='abierta')
    fecha_creacion = db.Column(db.String(40))
    fecha_cierre = db.Column(db.String(40), nullable=True)
    respuesta_admin = db.Column(db.Text, nullable=True)

    def __init__(self, vivienda_oid, comunidad_oid, usuario_oid,
                 titulo, descripcion, tipo='consulta'):
        self.vivienda_oid    = int(vivienda_oid)
        self.comunidad_oid   = int(comunidad_oid)
        self.usuario_oid     = int(usuario_oid)
        self.titulo          = titulo
        self.descripcion     = descripcion
        self.tipo            = tipo   # 'averia', 'consulta', 'otro'
        self.estado          = 'abierta'  # 'abierta', 'en_proceso', 'cerrada'
        self.fecha_creacion  = date.today().isoformat()
        self.fecha_cierre    = None
        self.respuesta_admin = None

    @property
    def __oid__(self):
        return self.id

    def __repr__(self):
        return f'<Incidencia {self.tipo} estado={self.estado} u={self.usuario_oid}>'
