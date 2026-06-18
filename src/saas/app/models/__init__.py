# Modelos de LeaLink
from app.extensions import db
from app.models.comunidad import Comunidad
from app.models.vivienda import Vivienda
from app.models.usuario import Usuario
from app.models.acceso import AccesoVivienda
from app.models.bateria import Bateria
from app.models.cierre import CierreMensual
from app.models.incidencia import Incidencia
from app.models.notificacion import Notificacion
from app.models.operacion import OperacionHoraria

__all__ = [
    'db', 'Comunidad', 'Vivienda', 'Usuario', 'AccesoVivienda',
    'Bateria', 'CierreMensual', 'Incidencia', 'Notificacion', 'OperacionHoraria'
]
