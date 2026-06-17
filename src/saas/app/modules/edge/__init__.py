from flask import Blueprint
edge_bp = Blueprint('edge', __name__)
from app.modules.edge import routes  # noqa
