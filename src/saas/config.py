"""Configuración de la aplicación LeaLink."""
import os
from dotenv import load_dotenv

load_dotenv()


def _normalizar_db_url(url: str) -> str:
    """Normaliza la URL de la BD.

    Algunos proveedores (Heroku) entregan el esquema legacy `postgres://`,
    que SQLAlchemy 2.x ya no acepta; se reescribe a `postgresql://`.
    """
    if url.startswith('postgres://'):
        return url.replace('postgres://', 'postgresql://', 1)
    return url


class BaseConfig:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key')
    # Base de datos: PostgreSQL en dev/prod (DATABASE_URL); SQLite por defecto en local.
    SQLALCHEMY_DATABASE_URI = _normalizar_db_url(
        os.environ.get('DATABASE_URL', 'sqlite:///lealink.db')
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    WTF_CSRF_ENABLED = True
    WTF_CSRF_HEADERS = ['X-CSRFToken', 'X-CSRF-Token']


class DevelopmentConfig(BaseConfig):
    DEBUG = True


class ProductionConfig(BaseConfig):
    DEBUG = False


class TestingConfig(BaseConfig):
    """SQLite en memoria, sin infraestructura, para los tests."""
    from sqlalchemy.pool import StaticPool

    TESTING = True
    DEBUG = False
    WTF_CSRF_ENABLED = False
    SQLALCHEMY_DATABASE_URI = 'sqlite://'
    # StaticPool + única conexión compartida → la BD en memoria es la misma
    # para todos los contextos de aplicación durante el test.
    SQLALCHEMY_ENGINE_OPTIONS = {
        'connect_args': {'check_same_thread': False},
        'poolclass': StaticPool,
    }


_configs = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
}


def get_config(nombre=None):
    nombre = nombre or os.environ.get('FLASK_ENV', 'development')
    return _configs.get(nombre, DevelopmentConfig)
