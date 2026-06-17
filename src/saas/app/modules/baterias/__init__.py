from flask import Blueprint
baterias_bp = Blueprint('baterias', __name__)
from app.modules.baterias import routes  # noqa
