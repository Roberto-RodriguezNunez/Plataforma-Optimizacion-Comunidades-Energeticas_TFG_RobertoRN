"""Extensiones compartidas de LeaLink.

Se aíslan aquí para evitar imports circulares entre la factory (app/__init__.py)
y los modelos (app/models/*).
"""
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()
