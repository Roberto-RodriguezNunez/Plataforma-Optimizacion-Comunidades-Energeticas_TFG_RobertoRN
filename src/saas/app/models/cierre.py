"""Modelo de CierreMensual de una vivienda."""
from datetime import datetime

from app.extensions import db


class CierreMensual(db.Model):
    """Cierre energético mensual de una vivienda.

    Contiene los datos de consumo, generación y ahorro de un mes concreto.
    Es la salida del agente RL del TFG; en esta versión se introduce manualmente.
    El par (vivienda_oid, mes) debe ser único.
    """

    __tablename__ = 'cierres'
    __table_args__ = (
        db.UniqueConstraint('vivienda_oid', 'mes', name='uq_cierre_vivienda_mes'),
    )

    id = db.Column(db.Integer, primary_key=True)
    vivienda_oid = db.Column(db.Integer, db.ForeignKey('viviendas.id'),
                             nullable=False, index=True)
    mes = db.Column(db.String(7), nullable=False)  # formato 'YYYY-MM'
    consumo_total_kwh = db.Column(db.Float, default=0.0)
    autoconsumo_directo_kwh = db.Column(db.Float, default=0.0)
    energia_de_bateria_kwh = db.Column(db.Float, default=0.0)
    vertido_a_red_kwh = db.Column(db.Float, default=0.0)
    ahorro_eur = db.Column(db.Float, default=0.0)
    factura_sin_paneles_eur = db.Column(db.Float, default=0.0)
    # Escenario 1: autoconsumo individual, sin comunidad ni batería
    factura_escenario_base_eur = db.Column(db.Float, default=0.0)
    # Escenario 2: autoconsumo colectivo entre viviendas, sin batería
    factura_paneles_comunidad_eur = db.Column(db.Float, default=0.0)
    # Escenario 3: autoconsumo colectivo + batería gestionada por SAC
    factura_escenario_real_eur = db.Column(db.Float, default=0.0)
    porcentaje_ahorro_global = db.Column(db.Float, default=0.0)
    coeficiente_reparto_aplicado = db.Column(db.Float, default=0.0)
    fecha_creacion = db.Column(db.String(40))

    def __init__(self, vivienda_oid, mes, consumo_total_kwh=0.0,
                 autoconsumo_directo_kwh=0.0, energia_de_bateria_kwh=0.0,
                 vertido_a_red_kwh=0.0, ahorro_eur=0.0,
                 factura_sin_paneles_eur=0.0,
                 factura_escenario_base_eur=0.0,
                 factura_paneles_comunidad_eur=0.0,
                 factura_escenario_real_eur=0.0,
                 porcentaje_ahorro_global=0.0, coeficiente_reparto_aplicado=0.0):
        self.vivienda_oid = int(vivienda_oid)
        self.mes = mes  # formato 'YYYY-MM'
        self.consumo_total_kwh = float(consumo_total_kwh)
        self.autoconsumo_directo_kwh = float(autoconsumo_directo_kwh)
        self.energia_de_bateria_kwh = float(energia_de_bateria_kwh)
        self.vertido_a_red_kwh = float(vertido_a_red_kwh)
        self.ahorro_eur = float(ahorro_eur)
        self.factura_sin_paneles_eur = float(factura_sin_paneles_eur)
        self.factura_escenario_base_eur = float(factura_escenario_base_eur)
        self.factura_paneles_comunidad_eur = float(factura_paneles_comunidad_eur)
        self.factura_escenario_real_eur = float(factura_escenario_real_eur)
        self.porcentaje_ahorro_global = float(porcentaje_ahorro_global)
        self.coeficiente_reparto_aplicado = float(coeficiente_reparto_aplicado)
        self.fecha_creacion = datetime.now().isoformat()

    @property
    def __oid__(self):
        return self.id

    @property
    def energia_de_red_kwh(self) -> float:
        """Energía consumida de la red eléctrica (resto tras autoconsumo y batería)."""
        return max(0.0, self.consumo_total_kwh
                   - self.autoconsumo_directo_kwh
                   - self.energia_de_bateria_kwh)

    def __repr__(self):
        return f'<CierreMensual {self.mes} v={self.vivienda_oid}>'
