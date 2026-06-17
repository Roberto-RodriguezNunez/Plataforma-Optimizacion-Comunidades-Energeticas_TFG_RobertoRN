from flask import Blueprint
cierres_bp = Blueprint('cierres', __name__)
from app.modules.cierres import routes  # noqa
