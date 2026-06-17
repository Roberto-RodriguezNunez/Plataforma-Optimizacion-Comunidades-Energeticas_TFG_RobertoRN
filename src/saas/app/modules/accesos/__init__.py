from flask import Blueprint
accesos_bp = Blueprint('accesos', __name__)
from app.modules.accesos import routes  # noqa
