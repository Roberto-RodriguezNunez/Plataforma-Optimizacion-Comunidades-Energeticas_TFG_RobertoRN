from flask import Blueprint
viviendas_bp = Blueprint('viviendas', __name__)
from app.modules.viviendas import routes  # noqa
