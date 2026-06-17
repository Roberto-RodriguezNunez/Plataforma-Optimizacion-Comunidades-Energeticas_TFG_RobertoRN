from flask import Blueprint
comunidades_bp = Blueprint('comunidades', __name__)
from app.modules.comunidades import routes  # noqa
