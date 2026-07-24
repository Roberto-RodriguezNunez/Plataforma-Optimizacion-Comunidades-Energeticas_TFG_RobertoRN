#!/usr/bin/env python3
"""
Seed de datos de ejemplo para LeaLink.
Idempotente: no hace nada si ya hay usuarios en la base de datos.

Genera:
  - 2 comunidades: Comunidade Solar de Vilarín (15 viviendas, O Courel) y Comunidade Enerxética de Brañas de Ulla (12 viviendas, Palas de Rei)
  - ~40 usuarios con nombres realistas
  - 1-3 usuarios por vivienda (titular obligatorio + conviventes opcionales)
  - La mayoría de usuarios tienen 1 vivienda; unos pocos tienen 2
  - Cierres mensuales de los últimos 3 meses por vivienda
  - Notificaciones de ejemplo
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import create_app
from app.extensions import db
from app.models.usuario import Usuario
from app.models.comunidad import Comunidad
from app.models.vivienda import Vivienda
from app.models.acceso import AccesoVivienda
from app.models.bateria import Bateria
from app.models.cierre import CierreMensual
from app.models.notificacion import Notificacion


# ---------------------------------------------------------------------------
# Datos base
# ---------------------------------------------------------------------------

MESES = ['2024-10', '2024-11', '2024-12']

# Nombres de usuarios con correos derivados del nombre
USUARIOS_DATA = [
    # (nombre, email, password)
    ('Roberto Rodríguez', 'roberto@lealink.es', 'roberto1234'),   # superadmin
    # Admins de comunidad
    ('Carmen Vidal Lago',     'carmen.vidal@vecinos.es',     'carmen1234'),
    ('Marcos Iglesias Otero', 'marcos.iglesias@vecinos.es',  'marcos1234'),
    # Resto de vecinos
    ('Ana García Suárez',      'ana.garcia@vecinos.es',      'ana1234'),
    ('Luis Fernández Rey',     'luis.fernandez@vecinos.es',  'luis1234'),
    ('María López Casal',      'maria.lopez@vecinos.es',     'maria1234'),
    ('José Martínez Díaz',     'jose.martinez@vecinos.es',   'jose1234'),
    ('Elena Pérez Ramos',      'elena.perez@vecinos.es',     'elena1234'),
    ('David González Soto',    'david.gonzalez@vecinos.es',  'david1234'),
    ('Sara Rodríguez Méndez',  'sara.rodriguez@vecinos.es',  'sara1234'),
    ('Pablo Álvarez Núñez',    'pablo.alvarez@vecinos.es',   'pablo1234'),
    ('Laura Blanco Castro',    'laura.blanco@vecinos.es',    'laura1234'),
    ('Carlos Ramos Vega',      'carlos.ramos@vecinos.es',    'carlos1234'),
    ('Inés Domínguez Pardo',   'ines.dominguez@vecinos.es',  'ines1234'),
    ('Rubén Torres Costas',    'ruben.torres@vecinos.es',    'ruben1234'),
    ('Nuria Varela Lamas',     'nuria.varela@vecinos.es',    'nuria1234'),
    ('Alejandro Silva Porto',  'alex.silva@vecinos.es',      'alex1234'),
    ('Sofía Castro Bravo',     'sofia.castro@vecinos.es',    'sofia1234'),
    ('Manuel Prieto Pazos',    'manuel.prieto@vecinos.es',   'manuel1234'),
    ('Cristina Muñoz Ares',    'cristina.munoz@vecinos.es',  'cristina1234'),
    ('Fernando López Rey',     'fernando.lopez@vecinos.es',  'fernando1234'),
    ('Marta Sánchez Doval',    'marta.sanchez@vecinos.es',   'marta1234'),
    ('Diego Herrero Cid',      'diego.herrero@vecinos.es',   'diego1234'),
    ('Patricia Jiménez Lago',  'patricia.jimenez@vecinos.es','patricia1234'),
    ('Andrés Moreno Gesto',    'andres.moreno@vecinos.es',   'andres1234'),
    ('Verónica Ruiz Esteve',   'veronica.ruiz@vecinos.es',   'veronica1234'),
    ('Javier Navarro Pons',    'javier.navarro@vecinos.es',  'javier1234'),
    ('Lucía Ortega Rial',      'lucia.ortega@vecinos.es',    'lucia1234'),
    ('Iván Molina Brea',       'ivan.molina@vecinos.es',     'ivan1234'),
    ('Beatriz Serrano Gil',    'beatriz.serrano@vecinos.es', 'beatriz1234'),
    ('Rafael Medina Pose',     'rafael.medina@vecinos.es',   'rafael1234'),
    ('Teresa Delgado Faro',    'teresa.delgado@vecinos.es',  'teresa1234'),
    ('Héctor Romero Montes',   'hector.romero@vecinos.es',   'hector1234'),
    ('Adriana Suárez Ledo',    'adriana.suarez@vecinos.es',  'adriana1234'),
    ('Gonzalo Reyes Sanz',     'gonzalo.reyes@vecinos.es',   'gonzalo1234'),
    ('Claudia Morales Pena',   'claudia.morales@vecinos.es', 'claudia1234'),
    ('Sergio Castro Vidal',    'sergio.castro@vecinos.es',   'sergio1234'),
    ('Pilar Guerrero Lage',    'pilar.guerrero@vecinos.es',  'pilar1234'),
    ('Tomás Reina Fuertes',    'tomas.reina@vecinos.es',     'tomas1234'),
    ('Amparo Nieto Sousa',     'amparo.nieto@vecinos.es',    'amparo1234'),
]

# Comunidade Solar de Vilarín — O Courel, Lugo (15 viviendas)
# Pueblo rural de montaña en la Serra do Courel (Lugo).
# Tres rutas: Rúa da Igrexa (6 casas), Camiño do Río (6 casas), Lugar de Outeiro (3 casas).
# Coeficientes proporcionales a la potencia contratada, suman exactamente 1.0.
# Todas con paneles (comunidad autoconsumo colectivo RD 244/2019). Panel estándar 400 Wp.
VIVIENDAS_VILARIN = [
    # (identificador, direccion, cups_suffix, potencia, coef, paneles, kwp, npaneles, orient)
    # kWp escala 42/54.4 = 0.772 sobre valores anteriores. Total: 42.0 kWp, 105 paneles 400 Wp.
    # Criterio: 15×3500 kWh/año ÷ 1250 h_eq/año (Galicia) = 42 kWp → gen ≈ consumo anual (RD 244/2019).
    ('Casa 1 — Rúa da Igrexa',   'Rúa da Igrexa 1', 'I001', 3.45, 0.0526, True, 2.4,  6, 'sur'),
    ('Casa 2 — Rúa da Igrexa',   'Rúa da Igrexa 3', 'I002', 4.60, 0.0702, True, 2.4,  6, 'sur'),
    ('Casa 3 — Rúa da Igrexa',   'Rúa da Igrexa 5', 'I003', 3.45, 0.0526, True, 2.0,  5, 'sur'),
    ('Casa 4 — Rúa da Igrexa',   'Rúa da Igrexa 7', 'I004', 3.45, 0.0526, True, 2.0,  5, 'sur'),
    ('Casa 5 — Rúa da Igrexa',   'Rúa da Igrexa 9', 'I005', 4.60, 0.0702, True, 3.2,  8, 'sur'),
    ('Casa 6 — Rúa da Igrexa',   'Rúa da Igrexa 11','I006', 3.45, 0.0526, True, 2.0,  5, 'sur'),
    ('Casa 1 — Camiño do Río',   'Camiño do Río 2', 'R001', 5.75, 0.0877, True, 4.4, 11, 'sur'),
    ('Casa 2 — Camiño do Río',   'Camiño do Río 4', 'R002', 4.60, 0.0702, True, 2.8,  7, 'sur'),
    ('Casa 3 — Camiño do Río',   'Camiño do Río 6', 'R003', 3.45, 0.0526, True, 2.0,  5, 'sur'),
    ('Casa 4 — Camiño do Río',   'Camiño do Río 8', 'R004', 4.60, 0.0702, True, 2.4,  6, 'sur'),
    ('Casa 5 — Camiño do Río',   'Camiño do Río 10','R005', 5.75, 0.0877, True, 5.2, 13, 'sur'),
    ('Casa 6 — Camiño do Río',   'Camiño do Río 12','R006', 4.60, 0.0702, True, 2.8,  7, 'sur'),
    ('Casa 1 — Lugar de Outeiro','Lugar de Outeiro 1','O001', 3.45, 0.0526, True, 2.4,  6, 'sur'),
    ('Casa 2 — Lugar de Outeiro','Lugar de Outeiro 3','O002', 4.60, 0.0702, True, 2.4,  6, 'sur'),
    ('Casa 3 — Lugar de Outeiro','Lugar de Outeiro 5','O003', 5.75, 0.0878, True, 3.6,  9, 'sur'),
]

# Comunidade Enerxética de Brañas de Ulla — Palas de Rei, Lugo (12 viviendas)
# Aldea en la comarca de Lugo próxima al Camino de Santiago.
# Dos zonas: Rúa do Muíño (6 casas), Lugar da Carballeira (6 casas).
# Coeficientes no están normalizados al 100% (comunidad en expansión).
VIVIENDAS_BRANAS = [
    # (identificador, direccion, cups_suffix, potencia, coef, paneles, kwp, npaneles, orient)
    ('Casa 1 — Rúa do Muíño',        'Rúa do Muíño 1',         'M001', 5.75, 0.090, True,  6.0, 15, 'sur'),
    ('Casa 2 — Rúa do Muíño',        'Rúa do Muíño 3',         'M002', 4.60, 0.082, False, None,None,None),
    ('Casa 3 — Rúa do Muíño',        'Rúa do Muíño 5',         'M003', 4.60, 0.082, True,  4.0, 10, 'sur'),
    ('Casa 4 — Rúa do Muíño',        'Rúa do Muíño 7',         'M004', 3.45, 0.078, False, None,None,None),
    ('Casa 5 — Rúa do Muíño',        'Rúa do Muíño 9',         'M005', 5.75, 0.090, True,  5.6, 14, 'sur'),
    ('Casa 6 — Rúa do Muíño',        'Rúa do Muíño 11',        'M006', 4.60, 0.082, False, None,None,None),
    ('Casa 1 — Lugar da Carballeira','Lugar da Carballeira 1',  'C001', 4.60, 0.082, True,  3.6,  9, 'sur'),
    ('Casa 2 — Lugar da Carballeira','Lugar da Carballeira 3',  'C002', 3.45, 0.078, False, None,None,None),
    ('Casa 3 — Lugar da Carballeira','Lugar da Carballeira 5',  'C003', 5.75, 0.090, True,  7.2, 18, 'sur'),
    ('Casa 4 — Lugar da Carballeira','Lugar da Carballeira 7',  'C004', 4.60, 0.082, False, None,None,None),
    ('Casa 5 — Lugar da Carballeira','Lugar da Carballeira 9',  'C005', 4.60, 0.082, True,  4.8, 12, 'sur'),
    ('Casa 6 — Lugar da Carballeira','Lugar da Carballeira 11', 'C006', 3.45, 0.082, False, None,None,None),
]

# Plan de accesos: (índice_vivienda_0based, índice_usuario_0based, rol)
# Usuarios 0=Roberto(superadmin), 1=Carmen, 2=Marcos, 3..39 = vecinos
#
# Asignación: cada vivienda tiene 1 titular + 0-2 conviventes
# La mayoría de usuarios salen una vez; unos pocos salen dos veces

ACCESOS_VIGO = [
    # vivienda_idx, usuario_idx, rol
    (0,  1,  'titular'),     # Carmen: titular V0
    (0,  3,  'convivente'),  # Ana convive con Carmen
    (1,  4,  'titular'),     # Luis
    (1,  5,  'convivente'),  # María convive con Luis
    (2,  6,  'titular'),     # José
    (2,  7,  'convivente'),  # Elena convive con José
    (2,  8,  'solo_lectura'),# David solo_lectura (hijo universitario)
    (3,  9,  'titular'),     # Sara
    (4,  10, 'titular'),     # Pablo
    (4,  11, 'convivente'),  # Laura convive con Pablo
    (5,  12, 'titular'),     # Carlos
    (6,  13, 'titular'),     # Inés
    (6,  14, 'convivente'),  # Rubén convive con Inés
    (7,  15, 'titular'),     # Nuria
    (7,  16, 'convivente'),  # Alejandro convive con Nuria
    (8,  17, 'titular'),     # Sofía
    (9,  18, 'titular'),     # Manuel
    (9,  19, 'convivente'),  # Cristina convive con Manuel
    (10, 20, 'titular'),     # Fernando
    (10, 21, 'convivente'),  # Marta convive con Fernando
    (11, 22, 'titular'),     # Diego
    (12, 23, 'titular'),     # Patricia
    (12, 24, 'convivente'),  # Andrés convive con Patricia
    (13, 25, 'titular'),     # Verónica
    (14, 26, 'titular'),     # Javier
    (14, 27, 'convivente'),  # Lucía convive con Javier
    # Iván tiene 2 viviendas (heredó una de su madre) — raro pero posible
    (3,  28, 'convivente'),  # Iván también en V3 con Sara
]

ACCESOS_SANTIAGO = [
    # vivienda_idx_dentro_de_Santiago, usuario_idx, rol
    (0,  2,  'titular'),    # Marcos: titular Casa1
    (0,  29, 'convivente'), # Beatriz convive con Marcos
    (1,  30, 'titular'),    # Rafael
    (1,  31, 'convivente'), # Teresa convive con Rafael
    (2,  32, 'titular'),    # Héctor
    (3,  33, 'titular'),    # Adriana
    (3,  34, 'convivente'), # Gonzalo convive con Adriana
    (4,  35, 'titular'),    # Claudia
    (4,  36, 'convivente'), # Sergio convive con Claudia
    (5,  37, 'titular'),    # Pilar
    (6,  38, 'titular'),    # Tomás
    (6,  39, 'convivente'), # Amparo convive con Tomás
    (7,  28, 'titular'),    # Iván: titular Casa8 Santiago (su 2ª vivienda)
    (8,  3,  'titular'),    # Ana: titular Casa9 Santiago (su 2ª vivienda — heredó)
    (9,  4,  'titular'),    # Luis: titular Casa10 Santiago (trabajo en Santiago)
    (10, 5,  'titular'),    # María: titular Casa11
    (10, 6,  'convivente'), # José también en Casa11
    (11, 7,  'titular'),    # Elena: titular Casa12
]


def coefs_kwp(viviendas_data):
    """Coeficientes de reducción de gasto: kWp aportado / kWp total de la comunidad.

    Las viviendas sin paneles tienen coef 0 (no aportan producción, no reciben
    parte del ahorro común). La suma es exactamente 1.0 sobre las que aportan.
    """
    total = sum((v[6] or 0.0) for v in viviendas_data if v[5])
    return [round((v[6] or 0.0) / total, 6) if v[5] else 0.0 for v in viviendas_data]


def cierres_para_comunidad(viv_oids, viviendas_data, coefs, mes, idx_offset=0):
    """Genera los cierres de un mes para toda la comunidad (modelo cooperativa).

    La producción se netea a nivel comunidad y el AHORRO total (frente a que
    cada casa fuera sola con sus paneles, sin batería) se resta a la factura
    de cada vecino según su coeficiente (kWp aportado). La suma de cuotas
    reproduce la factura común neteada.
    """
    m = MESES.index(mes)
    base_consumo = [310.0, 290.0, 275.0][m]   # más consumo en invierno gallego
    factor_solar = [0.28, 0.22, 0.18][m]      # oct > nov > dic por irradiación

    filas = []
    for idx, (viv_oid, vd, coef) in enumerate(zip(viv_oids, viviendas_data, coefs)):
        tiene_paneles = vd[5]
        variacion = (((idx + idx_offset) * 17) % 60) - 30  # -30..+30, determinista
        consumo = round(base_consumo + variacion, 1)
        autoconsumo = round(consumo * factor_solar, 1) if tiene_paneles else 0.0
        vertido = round(autoconsumo * 0.15, 1)

        factura_sin_paneles = round(consumo * 0.22, 2)
        # Sin comunidad: solo paneles propios, sin batería ni reparto
        kwh_red_solo = max(0.0, consumo - autoconsumo)
        surplus_solo = max(0.0, autoconsumo - consumo)
        factura_base = round(max(0.0, kwh_red_solo * 0.22 - surplus_solo * 0.065), 2)

        filas.append(dict(viv_oid=viv_oid, coef=coef, consumo=consumo,
                          autoconsumo=autoconsumo, vertido=vertido,
                          sin_paneles=factura_sin_paneles, base=factura_base))

    # Factura común neteada con batería: la batería comunitaria desplaza parte
    # del consumo de red a horas valle y aprovecha el excedente conjunto.
    bateria_total = round(sum(f['consumo'] for f in filas) * 0.14, 1)
    factura_real_com = 0.0
    for f in filas:
        bat_i = f['coef'] * bateria_total
        kwh_red = max(0.0, f['consumo'] - f['autoconsumo'] - bat_i)
        factura_real_com += kwh_red * 0.18 + (f['autoconsumo'] + bat_i) * 0.05
    ahorro_total = max(0.0, sum(f['base'] for f in filas) - factura_real_com)

    cierres = []
    for f in filas:
        ahorro = round(f['coef'] * ahorro_total, 2)
        cierres.append(CierreMensual(
            vivienda_oid=f['viv_oid'],
            mes=mes,
            consumo_total_kwh=f['consumo'],
            autoconsumo_directo_kwh=round(f['autoconsumo'] * 0.85, 1),
            autoconsumo_indirecto_kwh=round(f['autoconsumo'] * 0.15, 1),
            energia_de_bateria_kwh=round(f['coef'] * bateria_total, 1),
            vertido_a_red_kwh=f['vertido'],
            ahorro_eur=ahorro,
            factura_sin_paneles_eur=f['sin_paneles'],
            factura_escenario_base_eur=f['base'],
            factura_escenario_real_eur=round(f['base'] - ahorro, 2),
            porcentaje_ahorro_global=round(f['coef'] * 100, 1),
            coeficiente_reparto_aplicado=f['coef']
        ))
    return cierres


def seed():
    app = create_app()

    def save(obj):
        """Persiste el objeto y devuelve su id (equivalente al antiguo srp.save)."""
        db.session.add(obj)
        db.session.flush()
        return obj.id

    with app.app_context():
        if Usuario.query.count() > 0:
            print("⚠️  Ya hay datos. Seed omitido (usa 'docker compose down -v' para limpiar).")
            return

        print("🌱 Iniciando seed...")

        # Coeficientes de reducción de gasto: kWp aportado / kWp total
        coefs_vilarin = coefs_kwp(VIVIENDAS_VILARIN)
        coefs_branas  = coefs_kwp(VIVIENDAS_BRANAS)
        suma_vilarin = round(sum(coefs_vilarin), 4)
        suma_branas  = round(sum(coefs_branas), 4)
        print(f"   Coeficientes (kWp) Vilarín: {suma_vilarin} | Brañas: {suma_branas}")
        assert abs(suma_vilarin - 1.0) < 0.01, f"Coef Vilarín no suman 1: {suma_vilarin}"
        assert abs(suma_branas  - 1.0) < 0.01, f"Coef Brañas no suman 1: {suma_branas}"

        # ------------------------------------------------------------------
        # Usuarios
        # ------------------------------------------------------------------
        print(f"\n👤 Creando {len(USUARIOS_DATA)} usuarios...")
        usuarios_oids = []
        for i, (nombre, email, pwd) in enumerate(USUARIOS_DATA):
            rol = 'superadmin' if i == 0 else 'normal'
            u = Usuario(nombre, email, pwd, rol)
            oid = save(u)
            usuarios_oids.append(oid)
        print(f"   ✅ {len(usuarios_oids)} usuarios creados")

        # ------------------------------------------------------------------
        # Comunidades
        # ------------------------------------------------------------------
        vilarin = Comunidad(
            nombre='Comunidade Solar de Vilarín',
            ubicacion='O Courel, Lugo',
            fecha_constitucion='2022-03-15',
            estado='activa',
            descripcion='Comunidade de autoconsumo colectivo (RD 244/2019) no concello de O Courel. '
                        '15 vivendas rurais, todas con paneis fotovoltaicos de 400 Wp. '
                        'Batería compartida de 20 kWh. Coeficientes proporcionais á potencia contratada.'
        )
        vilarin_oid = save(vilarin)

        branas = Comunidad(
            nombre='Comunidade Enerxética de Brañas de Ulla',
            ubicacion='Palas de Rei, Lugo',
            fecha_constitucion='2023-04-01',
            estado='activa',
            descripcion='Proxecto piloto de comunidade enerxética na aldea de Brañas de Ulla, '
                        'próxima ao Camiño de Santiago. 12 vivendas con batería de 30 kWh.'
        )
        branas_oid = save(branas)
        print(f"\n🏘️  2 comunidades creadas")

        # ------------------------------------------------------------------
        # Viviendas — Vilarín
        # ------------------------------------------------------------------
        print(f"\n🏠 Creando viviendas Vilarín ({len(VIVIENDAS_VILARIN)})...")
        vilarin_viv_oids = []
        for i, (ident, dir_, cups_suf, pot, coef, paneles, kwp, npan, orient) in enumerate(VIVIENDAS_VILARIN):
            v = Vivienda(
                comunidad_oid=vilarin_oid,
                identificador=ident,
                direccion_completa=f'{dir_}, 27123 Vilarín (O Courel)',
                cups=f'ES002700000000{cups_suf}F',
                potencia_contratada_kw=pot,
                coeficiente_reparto=coefs_vilarin[i],
                fecha_alta='2022-04-01',
                tiene_paneles=paneles,
                potencia_pico_paneles_kwp=kwp,
                numero_paneles=npan,
                fecha_instalacion_paneles='2022-06-01' if paneles else None,
                orientacion_paneles=orient
            )
            oid = save(v)
            vilarin_viv_oids.append(oid)

        # ------------------------------------------------------------------
        # Viviendas — Brañas de Ulla
        # ------------------------------------------------------------------
        print(f"🏠 Creando viviendas Brañas ({len(VIVIENDAS_BRANAS)})...")
        branas_viv_oids = []
        for i, (ident, dir_, cups_suf, pot, coef, paneles, kwp, npan, orient) in enumerate(VIVIENDAS_BRANAS):
            v = Vivienda(
                comunidad_oid=branas_oid,
                identificador=ident,
                direccion_completa=f'{dir_}, 27200 Brañas de Ulla (Palas de Rei)',
                cups=f'ES002700000001{cups_suf}F',
                potencia_contratada_kw=pot,
                coeficiente_reparto=coefs_branas[i],
                fecha_alta='2023-05-01',
                tiene_paneles=paneles,
                potencia_pico_paneles_kwp=kwp,
                numero_paneles=npan,
                fecha_instalacion_paneles='2023-07-01' if paneles else None,
                orientacion_paneles=orient
            )
            oid = save(v)
            branas_viv_oids.append(oid)

        # ------------------------------------------------------------------
        # Baterías
        # ------------------------------------------------------------------
        save(Bateria(
            comunidad_oid=vilarin_oid,
            capacidad_nominal_kwh=80.0,
            capacidad_util_actual_kwh=80.0,  # batería nueva (renovada al cambiar a 80 kWh)
            ciclos_acumulados=487,
            fecha_instalacion='2022-06-01',
            fabricante='BYD',
            modelo='Battery-Box 80 kWh',
            estado='operativa'
        ))
        save(Bateria(
            comunidad_oid=branas_oid,
            capacidad_nominal_kwh=80.0,
            capacidad_util_actual_kwh=77.6,  # 203 ciclos → 3% degradación (ratio original)
            ciclos_acumulados=203,
            fecha_instalacion='2023-07-15',
            fabricante='BYD',
            modelo='Battery-Box 80 kWh',
            estado='operativa'
        ))
        print(f"\n🔋 2 baterías creadas")

        # ------------------------------------------------------------------
        # Accesos — Vigo
        # ------------------------------------------------------------------
        print(f"\n👥 Creando accesos...")
        n_accesos = 0
        for (viv_idx, usr_idx, rol) in ACCESOS_VIGO:
            a = AccesoVivienda(
                usuario_oid=usuarios_oids[usr_idx],
                vivienda_oid=vilarin_viv_oids[viv_idx],
                rol_en_vivienda=rol,
                fecha_incorporacion='2022-04-01'
            )
            save(a)
            n_accesos += 1

        # Accesos — Brañas de Ulla
        for (viv_idx, usr_idx, rol) in ACCESOS_SANTIAGO:
            a = AccesoVivienda(
                usuario_oid=usuarios_oids[usr_idx],
                vivienda_oid=branas_viv_oids[viv_idx],
                rol_en_vivienda=rol,
                fecha_incorporacion='2023-05-01'
            )
            save(a)
            n_accesos += 1

        print(f"   ✅ {n_accesos} accesos creados")

        # ------------------------------------------------------------------
        # Cierres mensuales
        # ------------------------------------------------------------------
        # Vilarín (comunidad 1) la alimenta el edge en vivo, así que NO se siembran
        # sus cierres de 2024 para no mezclarlos con los que publica el edge.
        # Brañas mantiene los cierres de demostración de los 3 meses de 2024.
        print(f"\n📊 Creando cierres mensuales de Brañas ({len(MESES)} meses × {len(VIVIENDAS_BRANAS)} viviendas)...")
        n_cierres = 0
        for mes in MESES:
            for c in cierres_para_comunidad(branas_viv_oids, VIVIENDAS_BRANAS,
                                            coefs_branas, mes, idx_offset=20):
                save(c)
                n_cierres += 1

        print(f"   ✅ {n_cierres} cierres creados")

        # ------------------------------------------------------------------
        # Notificaciones de ejemplo
        # ------------------------------------------------------------------
        print(f"\n🔔 Creando notificaciones...")

        def notif(usr_idx, tipo, titulo, mensaje, leida=False):
            n = Notificacion(usuarios_oids[usr_idx], tipo, titulo, mensaje)
            n.leida = leida
            save(n)

        # Para Carmen (admin Vilarín)
        notif(1, 'general', '¡Bienvenida, administradora!',
              'Tu cuenta de administradora de Comunidade Solar de Vilarín está activa.', leida=True)
        notif(1, 'cierre_disponible', 'Cierres publicados',
              'Los cierres energéticos ya están disponibles para las viviendas de tu comunidad.')
        notif(1, 'bateria_mantenimiento', 'Revisión anual de batería programada',
              'La batería BYD Battery-Box 80 kWh tiene programado su mantenimiento anual para enero de 2025.')

        # Para Marcos (admin Brañas de Ulla)
        notif(2, 'general', '¡Bienvenido, administrador!',
              'Tu cuenta de administrador de Comunidade Enerxética de Brañas de Ulla está activa.', leida=True)
        notif(2, 'cierre_disponible', 'Cierres de diciembre publicados',
              'Os peches de 2024-12 xa están dispoñibles para todas as vivendas.')
        notif(2, 'incidencia', 'Incidencia en inversor resuelta',
              'O fallo detectado no inversor do Lugar da Carballeira foi corrixido polo técnico.')

        # Para usuarios normales (los primeros 6 vecinos)
        for usr_idx, nombre_corto in [(3,'Ana'), (4,'Luis'), (5,'María'), (6,'José'), (9,'Sara'), (10,'Pablo')]:
            notif(usr_idx, 'cierre_disponible', f'Cierre disponible',
                  f'El cierre energético más reciente de tu vivienda ya está disponible. '
                  f'Consulta tu ahorro en el detalle de vivienda.')

        # Una no leída para demostrar el contador
        notif(3, 'cambio_coeficiente', 'Actualización de coeficientes',
              'Los coeficientes de reparto han sido revisados para 2025. Tu coeficiente no varía.')

        # Confirmar toda la transacción en PostgreSQL
        db.session.commit()

        print(f"   ✅ Notificaciones creadas")

        # ------------------------------------------------------------------
        # Resumen
        # ------------------------------------------------------------------
        print(f"""
╔══════════════════════════════════════════════════════╗
║           ✨ Seed completado correctamente           ║
╠══════════════════════════════════════════════════════╣
║  Usuarios:   {len(usuarios_oids):3}   Viviendas: {len(vilarin_viv_oids)+len(branas_viv_oids):3}         ║
║  Accesos:    {n_accesos:3}   Cierres:   {n_cierres:3}         ║
╠══════════════════════════════════════════════════════╣
║  Credenciales:                                       ║
║  roberto@lealink.es     /  roberto1234  (superadmin) ║
║  carmen.vidal@vecinos.es / carmen1234 (admin Vilarín) ║
║  marcos.iglesias@vecinos.es / marcos1234 (admin Brañas)║
║  ana.garcia@vecinos.es  /  ana1234      (vecina)     ║
╚══════════════════════════════════════════════════════╝
""")


if __name__ == '__main__':
    seed()
