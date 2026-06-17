from flask import Blueprint
usuarios_bp = Blueprint('usuarios', __name__)
from app.modules.usuarios import routes  # noqa
