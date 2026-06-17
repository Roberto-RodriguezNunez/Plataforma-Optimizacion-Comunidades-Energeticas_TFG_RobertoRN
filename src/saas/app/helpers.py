"""
Utilidades centrales de LeaLink.
Importado por todos los blueprints — mantener sin imports circulares.
"""
from flask import flash, request as _request

from app.extensions import db


# ---------------------------------------------------------------------------
# IDs URL-safe
#
# Tras la migración a PostgreSQL, los identificadores son enteros (PK). Se
# conservan estas funciones (y el atributo `__oid__` de los modelos) para no
# tener que reescribir rutas ni plantillas: el "safe oid" es simplemente el id
# en texto.
# ---------------------------------------------------------------------------

def oid_to_safe(oid) -> str:
    """Convierte un id de modelo a string apto para rutas Flask."""
    return str(oid)


def oid_from_safe(safe: str) -> int:
    """Reconstruye un id entero desde un string de URL.

    Lanza ValueError si el formato es inválido (las rutas lo capturan → 404).
    """
    return int(safe)


# ---------------------------------------------------------------------------
# Detección de petición AJAX
# ---------------------------------------------------------------------------

def is_xhr(req=None) -> bool:
    """Devuelve True si la petición es XMLHttpRequest (AJAX)."""
    req = req or _request
    return req.headers.get('X-Requested-With') == 'XMLHttpRequest'


# ---------------------------------------------------------------------------
# Flash helpers
# ---------------------------------------------------------------------------

def flash_exito(mensaje: str):
    flash(mensaje, 'success')


def flash_error(mensaje: str):
    flash(mensaje, 'danger')


# ---------------------------------------------------------------------------
# Comprobaciones de permisos
# ---------------------------------------------------------------------------

def usuario_tiene_acceso(usuario_oid, vivienda_oid) -> bool:
    """Comprueba si el usuario tiene algún acceso a la vivienda."""
    from app.models.acceso import AccesoVivienda

    return AccesoVivienda.query.filter_by(
        usuario_oid=int(usuario_oid), vivienda_oid=int(vivienda_oid)
    ).first() is not None


# ---------------------------------------------------------------------------
# Recálculo de coeficientes de reparto
# ---------------------------------------------------------------------------

def recalcular_coeficientes(comunidad_oid):
    """Recalcula los coeficientes de reparto de todas las viviendas de una
    comunidad, proporcionales a su potencia contratada.

    coef_i = potencia_i / suma_potencias
    """
    from app.models.vivienda import Vivienda
    from app.models.acceso import AccesoVivienda

    com_id = int(comunidad_oid)
    viviendas = Vivienda.query.filter_by(comunidad_oid=com_id).all()
    if not viviendas:
        return
    suma = sum(v.potencia_contratada_kw for v in viviendas)
    if suma <= 0:
        return

    for v in viviendas:
        nuevo = round(v.potencia_contratada_kw / suma, 6)
        cambio = v.coeficiente_reparto != nuevo
        v.coeficiente_reparto = nuevo

        if cambio:
            accesos = AccesoVivienda.query.filter_by(vivienda_oid=v.id).all()
            usuarios = {a.usuario_oid for a in accesos}
            for usr_oid in usuarios:
                crear_notificacion(
                    usr_oid, 'cambio_coeficiente',
                    f'Coeficiente actualizado — {v.identificador}',
                    f'Tu coeficiente de reparto en {v.identificador} ha cambiado a {nuevo:.4f}.',
                )
    db.session.commit()


# ---------------------------------------------------------------------------
# Borrado en cascada
# ---------------------------------------------------------------------------

def cascade_delete_vivienda(vivienda_oid):
    """Borra una vivienda y todos sus objetos dependientes.

    Cascada: AccesoVivienda → CierreMensual → Incidencia → Vivienda
    """
    from app.models.acceso import AccesoVivienda
    from app.models.cierre import CierreMensual
    from app.models.incidencia import Incidencia
    from app.models.vivienda import Vivienda

    viv_id = int(vivienda_oid)

    AccesoVivienda.query.filter_by(vivienda_oid=viv_id).delete()
    CierreMensual.query.filter_by(vivienda_oid=viv_id).delete()
    Incidencia.query.filter_by(vivienda_oid=viv_id).delete()

    viv = db.session.get(Vivienda, viv_id)
    if viv:
        db.session.delete(viv)
    db.session.commit()


def cascade_delete_comunidad(comunidad_oid):
    """Borra una comunidad y todos sus objetos dependientes.

    Cascada: Viviendas (con su propia cascada) → Bateria → Notificaciones relacionadas → Comunidad
    """
    from app.models.vivienda import Vivienda
    from app.models.bateria import Bateria
    from app.models.notificacion import Notificacion
    from app.models.comunidad import Comunidad

    com_id = int(comunidad_oid)

    # Borrar cada vivienda con su cascada
    for v in Vivienda.query.filter_by(comunidad_oid=com_id).all():
        cascade_delete_vivienda(v.id)

    # Borrar batería
    Bateria.query.filter_by(comunidad_oid=com_id).delete()

    # Borrar notificaciones relacionadas con esta comunidad
    Notificacion.query.filter_by(entidad_relacionada_oid=com_id).delete()

    com = db.session.get(Comunidad, com_id)
    if com:
        db.session.delete(com)
    db.session.commit()


def puede_borrar_usuario(usuario_oid) -> tuple:
    """Comprueba si es seguro borrar un usuario.

    Devuelve (True, '') si se puede borrar.
    Devuelve (False, motivo) si hay algún bloqueo.

    Bloqueos:
    - Es el único titular de alguna vivienda.
    """
    from app.models.acceso import AccesoVivienda
    from app.models.vivienda import Vivienda

    usr_id = int(usuario_oid)
    accesos_usuario = AccesoVivienda.query.filter_by(usuario_oid=usr_id).all()

    for acceso in accesos_usuario:
        if acceso.rol_en_vivienda == 'titular':
            otros_titulares = AccesoVivienda.query.filter(
                AccesoVivienda.vivienda_oid == acceso.vivienda_oid,
                AccesoVivienda.rol_en_vivienda == 'titular',
                AccesoVivienda.usuario_oid != usr_id,
            ).first()
            if not otros_titulares:
                viv = db.session.get(Vivienda, acceso.vivienda_oid)
                nombre_viv = viv.identificador if viv else acceso.vivienda_oid
                return False, f'Es el único titular de la vivienda "{nombre_viv}"'

    return True, ''


# ---------------------------------------------------------------------------
# Creación de notificaciones
# ---------------------------------------------------------------------------

def crear_notificacion(usuario_oid, tipo, titulo, mensaje,
                       entidad_oid=None, entidad_tipo=None):
    """Crea y guarda una notificación para un usuario."""
    from app.models.notificacion import Notificacion
    n = Notificacion(
        usuario_oid=int(usuario_oid),
        tipo=tipo,
        titulo=titulo,
        mensaje=mensaje,
        entidad_relacionada_oid=entidad_oid,
        entidad_relacionada_tipo=entidad_tipo
    )
    db.session.add(n)
    db.session.commit()
    return n


def notificar_a_comunidad(comunidad_oid, tipo, titulo, mensaje,
                           entidad_oid=None, entidad_tipo=None):
    """Envía una notificación a todos los usuarios con acceso a la comunidad."""
    from app.models.vivienda import Vivienda
    from app.models.acceso import AccesoVivienda

    com_id = int(comunidad_oid)
    viviendas_ids = [
        v.id for v in Vivienda.query.filter_by(comunidad_oid=com_id).all()
    ]
    if not viviendas_ids:
        return
    # Usuarios únicos con acceso
    accesos = AccesoVivienda.query.filter(
        AccesoVivienda.vivienda_oid.in_(viviendas_ids)
    ).all()
    usuarios_notificados = {a.usuario_oid for a in accesos}
    for usr in usuarios_notificados:
        crear_notificacion(usr, tipo, titulo, mensaje, entidad_oid, entidad_tipo)
