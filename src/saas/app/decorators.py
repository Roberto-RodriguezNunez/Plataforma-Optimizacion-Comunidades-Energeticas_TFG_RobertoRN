"""Decoradores de autorización para LeaLink."""
from functools import wraps
from flask import abort
from flask_login import current_user


def superadmin_required(f):
    """Exige que el usuario sea superadmin del SaaS."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if not current_user.is_authenticated:
            abort(401)
        if current_user.rol_global != 'superadmin':
            abort(403)
        return f(*args, **kwargs)
    return decorated


def acceso_vivienda_required(param='safe_oid'):
    """Exige que el usuario tenga algún acceso a la vivienda indicada."""
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            if not current_user.is_authenticated:
                abort(401)
            if current_user.rol_global == 'superadmin':
                return f(*args, **kwargs)
            from app.helpers import usuario_tiene_acceso, oid_from_safe
            safe_oid = kwargs.get(param)
            if not safe_oid:
                abort(403)
            try:
                vivienda_oid = oid_from_safe(safe_oid)
            except Exception:
                abort(404)
            if not usuario_tiene_acceso(current_user.id, vivienda_oid):
                abort(403)
            return f(*args, **kwargs)
        return decorated
    return decorator
