"""Modelo de Usuario de la plataforma."""
from datetime import datetime
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

from app.extensions import db


class Usuario(UserMixin, db.Model):
    """Usuario de LeaLink.

    Puede ser superadmin del SaaS o usuario normal.
    Flask-Login usa get_id() para serializar la sesión.
    """

    __tablename__ = 'usuarios'

    ROLES = ('superadmin', 'normal')

    id = db.Column(db.Integer, primary_key=True)
    nombre = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(255), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    rol_global = db.Column(db.String(20), nullable=False, default='normal')
    fecha_registro = db.Column(db.String(40))

    def __init__(self, nombre, email, password, rol_global='normal'):
        self.nombre = nombre
        self.email = email.lower().strip()
        self.password_hash = generate_password_hash(password)
        self.rol_global = rol_global
        self.fecha_registro = datetime.now().isoformat()

    @property
    def __oid__(self):
        """Compatibilidad con el código que usaba OIDs de Sirope."""
        return self.id

    def verificar_password(self, password) -> bool:
        """Comprueba la contraseña en texto plano contra el hash almacenado."""
        return check_password_hash(self.password_hash, password)

    def set_password(self, nueva_password):
        """Actualiza el hash de contraseña."""
        self.password_hash = generate_password_hash(nueva_password)

    def get_id(self) -> str:
        """Flask-Login: devuelve identificador para la sesión."""
        return str(self.id)

    @property
    def es_superadmin(self) -> bool:
        return self.rol_global == 'superadmin'

    def __repr__(self):
        return f'<Usuario {self.email}>'
