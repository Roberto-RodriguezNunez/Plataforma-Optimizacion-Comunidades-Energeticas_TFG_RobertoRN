from flask import Blueprint
notificaciones_bp = Blueprint('notificaciones', __name__)
from app.modules.notificaciones import routes  # noqa
