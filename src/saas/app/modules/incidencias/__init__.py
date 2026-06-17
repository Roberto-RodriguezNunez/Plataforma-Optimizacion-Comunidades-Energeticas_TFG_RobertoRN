from flask import Blueprint
incidencias_bp = Blueprint('incidencias', __name__, url_prefix='/incidencias')
from app.modules.incidencias import routes  # noqa
