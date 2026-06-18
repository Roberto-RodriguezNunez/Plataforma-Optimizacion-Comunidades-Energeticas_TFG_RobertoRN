"""Modelo de Notificación para usuarios de la plataforma."""
from datetime import datetime

from app.extensions import db


class Notificacion(db.Model):
    """Notificación enviada a un usuario.

    Puede estar relacionada con una entidad concreta (batería, cierre, etc.).
    El usuario puede marcarla como leída.
    """

    __tablename__ = 'notificaciones'

    TIPOS = ('cierre_disponible', 'cambio_coeficiente', 'bateria_mantenimiento',
             'incidencia', 'general')

    id = db.Column(db.Integer, primary_key=True)
    usuario_oid = db.Column(db.Integer, db.ForeignKey('usuarios.id'),
                            nullable=False, index=True)
    fecha = db.Column(db.String(40))
    tipo = db.Column(db.String(30), nullable=False)
    titulo = db.Column(db.String(200), nullable=False)
    mensaje = db.Column(db.Text, nullable=False)
    # Referencia polimórfica (puede apuntar a comunidad, batería, cierre…): sin FK.
    entidad_relacionada_oid = db.Column(db.Integer, nullable=True)
    entidad_relacionada_tipo = db.Column(db.String(40), nullable=True)
    leida = db.Column(db.Boolean, nullable=False, default=False)

    def __init__(self, usuario_oid, tipo, titulo, mensaje,
                 entidad_relacionada_oid=None, entidad_relacionada_tipo=None):
        self.usuario_oid = int(usuario_oid)
        self.fecha = datetime.now().isoformat()
        self.tipo = tipo
        self.titulo = titulo
        self.mensaje = mensaje
        self.entidad_relacionada_oid = (
            int(entidad_relacionada_oid) if entidad_relacionada_oid else None
        )
        self.entidad_relacionada_tipo = entidad_relacionada_tipo
        self.leida = False

    @property
    def __oid__(self):
        return self.id

    def marcar_leida(self):
        self.leida = True

    def __repr__(self):
        return f'<Notificacion {self.tipo} u={self.usuario_oid} leida={self.leida}>'
