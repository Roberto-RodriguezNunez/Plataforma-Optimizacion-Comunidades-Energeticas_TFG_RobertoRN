from flask import Blueprint
auth_bp = Blueprint('auth', __name__)
from app.modules.auth import routes  # noqa
