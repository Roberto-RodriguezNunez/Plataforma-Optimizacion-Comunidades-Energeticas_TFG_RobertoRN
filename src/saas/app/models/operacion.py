"""Modelo de operación horaria registrada por el controlador edge."""
from datetime import datetime

from app.extensions import db


class OperacionHoraria(db.Model):
    """Registro de una decisión horaria del controlador ONNX edge.

    Una fila por hora y comunidad. Almacena tanto los datos del DataFeed
    (necesarios para calcular el ahorro real por vivienda en el cierre mensual)
    como la decisión tomada por el controlador y el beneficio resultante.
    """

    __tablename__ = 'operaciones_horarias'

    id            = db.Column(db.Integer, primary_key=True)
    comunidad_oid = db.Column(db.Integer, db.ForeignKey('comunidades.id'),
                              nullable=False, index=True)
    ts            = db.Column(db.DateTime, nullable=False, index=True,
                              default=datetime.utcnow)
    step          = db.Column(db.Integer)

    # Snapshot del DataFeed para esa hora (necesario para cierre mensual)
    consumo_total_kwh = db.Column(db.Float)
    gen_total_kwh     = db.Column(db.Float)
    precio_compra     = db.Column(db.Float)   # EUR/kWh
    precio_exc        = db.Column(db.Float)   # EUR/kWh excedente

    # Decisión del controlador edge
    soc               = db.Column(db.Float)   # 0..1
    p_carga_solar     = db.Column(db.Float)
    p_carga_red       = db.Column(db.Float)
    p_descarga_casa   = db.Column(db.Float)
    p_descarga_red    = db.Column(db.Float)
    beneficio_marginal = db.Column(db.Float)

    # Relación
    comunidad = db.relationship('Comunidad', backref=db.backref(
        'operaciones', lazy='dynamic', order_by='OperacionHoraria.ts'
    ))

    def __repr__(self):
        return (f'<OperacionHoraria comunidad={self.comunidad_oid} '
                f'step={self.step} soc={self.soc:.2f}>')
