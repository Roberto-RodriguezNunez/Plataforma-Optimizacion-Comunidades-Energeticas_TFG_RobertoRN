"""API endpoint para recibir decisiones horarias del controlador edge (ONNX/RPi).

Autenticación: header  Authorization: Bearer <EDGE_API_KEY>
La clave se configura con la variable de entorno EDGE_API_KEY en el SaaS.
"""
import os
from datetime import datetime, timezone

from flask import jsonify, request

from app.extensions import db
from app.models.bateria import Bateria
from app.models.operacion import OperacionHoraria
from app.modules.edge import edge_bp

_REQUIRED_FIELDS = {
    'comunidad_id', 'step',
    'consumo_total_kwh', 'gen_total_kwh', 'precio_compra', 'precio_exc',
    'soc', 'P_carga_solar', 'P_carga_red', 'P_descarga_casa', 'P_descarga_red',
    'beneficio_marginal',
}


def _check_auth():
    """Devuelve True si la clave API es válida o no se ha configurado."""
    api_key = os.environ.get('EDGE_API_KEY', '')
    if not api_key:
        return True  # sin clave configurada → no se exige (entorno dev)
    header = request.headers.get('Authorization', '')
    return header == f'Bearer {api_key}'


@edge_bp.route('/decision', methods=['POST'])
def recibir_decision():
    """Almacena una decisión horaria del edge y actualiza el SoC de la batería."""
    if not _check_auth():
        return jsonify({'error': 'No autorizado'}), 401

    data = request.get_json(silent=True)
    if not data:
        return jsonify({'error': 'Body JSON requerido'}), 400

    missing = _REQUIRED_FIELDS - data.keys()
    if missing:
        return jsonify({'error': f'Campos requeridos: {missing}'}), 400

    ts_raw = data.get('ts')
    ts = datetime.fromisoformat(ts_raw) if ts_raw else datetime.now(timezone.utc)

    op = OperacionHoraria(
        comunidad_oid=int(data['comunidad_id']),
        ts=ts,
        step=int(data['step']),
        consumo_total_kwh=float(data['consumo_total_kwh']),
        gen_total_kwh=float(data['gen_total_kwh']),
        precio_compra=float(data['precio_compra']),
        precio_exc=float(data['precio_exc']),
        soc=float(data['soc']),
        p_carga_solar=float(data['P_carga_solar']),
        p_carga_red=float(data['P_carga_red']),
        p_descarga_casa=float(data['P_descarga_casa']),
        p_descarga_red=float(data['P_descarga_red']),
        beneficio_marginal=float(data['beneficio_marginal']),
    )
    db.session.add(op)

    # Actualizar soc_actual en la batería de la comunidad
    bat = Bateria.query.filter_by(comunidad_oid=op.comunidad_oid).first()
    if bat:
        bat.soc_actual = op.soc

    db.session.commit()
    return jsonify({'ok': True, 'id': op.id}), 201


@edge_bp.route('/generar-cierre', methods=['POST'])
def generar_cierre_edge():
    """Genera el CierreMensual de un mes completo a partir de OperacionHoraria.

    Llamado automáticamente por el edge al cruzar el límite de mes.
    Autenticación: mismo Bearer token que /decision.
    Body JSON: { "comunidad_id": 1, "mes": "2025-01" }
    """
    if not _check_auth():
        return jsonify({'error': 'No autorizado'}), 401

    data = request.get_json(silent=True) or {}
    comunidad_id = data.get('comunidad_id')
    mes = data.get('mes')
    if not comunidad_id or not mes:
        return jsonify({'error': 'comunidad_id y mes requeridos'}), 400

    from app.modules.cierres.routes import _calcular_cierre_desde_operaciones
    n, msg = _calcular_cierre_desde_operaciones(int(comunidad_id), str(mes))
    status = 201 if n > 0 else 400
    return jsonify({'ok': n > 0, 'n_viviendas': n, 'msg': msg}), status
