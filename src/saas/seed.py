#!/usr/bin/env python3
"""
Seed de datos de ejemplo para LeaLink.
Idempotente: no hace nada si ya hay usuarios en la base de datos.

Genera:
  - 2 comunidades: Residencial Vigo Centro (15 viviendas) y Eco-Barrio Santiago (12 viviendas)
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

# Viviendas de Residencial Vigo Centro (15 viviendas)
# coeficientes: suman exactamente 1.00
VIVIENDAS_VIGO = [
    # (identificador, direccion, cups_suffix, potencia, coef, paneles, kwp, npaneles, orient)
    # Coeficientes proporcionales a la potencia contratada, suman exactamente 1.0
    # Todas las viviendas participan con paneles (comunidad de autoconsumo colectivo RD 244/2019)
    # kwp ≈ 0.7–1.0× potencia contratada; numero_paneles = round(kwp / 0.36)
    ('Portal A - 1ºIzq', 'Rúa do Príncipe 12, 1ºIzq', 'A001', 3.45, 0.0526, True, 3.2,  9, 'sur'),
    ('Portal A - 1ºDch', 'Rúa do Príncipe 12, 1ºDch', 'A002', 4.60, 0.0702, True, 3.2,  9, 'sur'),
    ('Portal A - 2ºIzq', 'Rúa do Príncipe 12, 2ºIzq', 'A003', 3.45, 0.0526, True, 2.8,  8, 'sur'),
    ('Portal A - 2ºDch', 'Rúa do Príncipe 12, 2ºDch', 'A004', 3.45, 0.0526, True, 2.4,  7, 'mixta'),
    ('Portal A - 3ºIzq', 'Rúa do Príncipe 12, 3ºIzq', 'A005', 4.60, 0.0702, True, 4.0, 11, 'mixta'),
    ('Portal A - 3ºDch', 'Rúa do Príncipe 12, 3ºDch', 'A006', 3.45, 0.0526, True, 2.8,  8, 'este'),
    ('Portal B - 1ºIzq', 'Rúa do Príncipe 14, 1ºIzq', 'B001', 5.75, 0.0877, True, 5.6, 16, 'sur'),
    ('Portal B - 1ºDch', 'Rúa do Príncipe 14, 1ºDch', 'B002', 4.60, 0.0702, True, 3.6, 10, 'sur'),
    ('Portal B - 2ºIzq', 'Rúa do Príncipe 14, 2ºIzq', 'B003', 3.45, 0.0526, True, 2.4,  7, 'este'),
    ('Portal B - 2ºDch', 'Rúa do Príncipe 14, 2ºDch', 'B004', 4.60, 0.0702, True, 3.2,  9, 'mixta'),
    ('Portal B - 3ºIzq', 'Rúa do Príncipe 14, 3ºIzq', 'B005', 5.75, 0.0877, True, 6.4, 18, 'sur'),
    ('Portal B - 3ºDch', 'Rúa do Príncipe 14, 3ºDch', 'B006', 4.60, 0.0702, True, 3.6, 10, 'sur'),
    ('Portal C - 1ºIzq', 'Rúa do Príncipe 16, 1ºIzq', 'C001', 3.45, 0.0526, True, 3.2,  9, 'sur'),
    ('Portal C - 1ºDch', 'Rúa do Príncipe 16, 1ºDch', 'C002', 4.60, 0.0702, True, 3.2,  9, 'sur'),
    ('Portal C - 2ºIzq', 'Rúa do Príncipe 16, 2ºIzq', 'C003', 5.75, 0.0878, True, 4.8, 13, 'sur'),
]

# Viviendas de Eco-Barrio Santiago (12 viviendas)
VIVIENDAS_SANTIAGO = [
    ('Casa 1 — Bloque A', 'Rúa das Hortas 3, Bloque A', 'H001', 5.75, 0.090, True,  6.0, 15, 'sur'),
    ('Casa 2 — Bloque A', 'Rúa das Hortas 3, Bloque A', 'H002', 4.60, 0.082, False, None,None,None),
    ('Casa 3 — Bloque A', 'Rúa das Hortas 3, Bloque A', 'H003', 4.60, 0.082, True,  4.0, 10, 'sur'),
    ('Casa 4 — Bloque A', 'Rúa das Hortas 3, Bloque A', 'H004', 3.45, 0.078, False, None,None,None),
    ('Casa 5 — Bloque B', 'Rúa das Hortas 5, Bloque B', 'H005', 5.75, 0.090, True,  5.6, 14, 'mixta'),
    ('Casa 6 — Bloque B', 'Rúa das Hortas 5, Bloque B', 'H006', 4.60, 0.082, False, None,None,None),
    ('Casa 7 — Bloque B', 'Rúa das Hortas 5, Bloque B', 'H007', 4.60, 0.082, True,  3.6, 9,  'sur'),
    ('Casa 8 — Bloque B', 'Rúa das Hortas 5, Bloque B', 'H008', 3.45, 0.078, False, None,None,None),
    ('Casa 9 — Bloque C', 'Rúa das Hortas 7, Bloque C', 'H009', 5.75, 0.090, True,  7.2, 18, 'sur'),
    ('Casa 10 — Bloque C','Rúa das Hortas 7, Bloque C', 'H010', 4.60, 0.082, False, None,None,None),
    ('Casa 11 — Bloque C','Rúa das Hortas 7, Bloque C', 'H011', 4.60, 0.082, True,  4.8, 12, 'este'),
    ('Casa 12 — Bloque C','Rúa das Hortas 7, Bloque C', 'H012', 3.45, 0.082, False, None,None,None),
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


def coef_suma(viviendas_data):
    """Verifica que los coeficientes sumen aproximadamente 1.0."""
    total = sum(v[4] for v in viviendas_data)
    return round(total, 4)


def cierre_para_vivienda(viv_oid, mes, coef, idx):
    """Genera datos de cierre históricos coherentes (todas las viviendas tienen paneles)."""
    # Base de consumo varía por mes (más en invierno gallego)
    base_consumo = [310.0, 290.0, 275.0][MESES.index(mes)]
    variacion = ((idx * 17) % 60) - 30  # -30..+30, determinista
    consumo = round(base_consumo + variacion, 1)

    # Generación solar (oct > nov > dic por irradiación)
    factor_solar = [0.28, 0.22, 0.18][MESES.index(mes)]
    autoconsumo = round(consumo * factor_solar, 1)
    bateria = round(consumo * 0.14, 1)
    vertido = round(autoconsumo * 0.15, 1)

    kwh_de_red = max(0.0, consumo - autoconsumo - bateria)
    factura_base = round(consumo * 0.22, 2)
    factura_real = round(kwh_de_red * 0.18 + (autoconsumo + bateria) * 0.05, 2)
    ahorro = round(max(0.0, factura_base - factura_real), 2)
    if ahorro == 0.0:
        factura_real = factura_base

    return CierreMensual(
        vivienda_oid=viv_oid,
        mes=mes,
        consumo_total_kwh=consumo,
        autoconsumo_directo_kwh=autoconsumo,
        energia_de_bateria_kwh=bateria,
        vertido_a_red_kwh=vertido,
        ahorro_eur=ahorro,
        factura_escenario_base_eur=factura_base,
        factura_escenario_real_eur=factura_real,
        porcentaje_ahorro_global=round(coef * 100, 1),
        coeficiente_reparto_aplicado=coef
    )


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

        # Verificar coherencia de coeficientes
        suma_vigo = coef_suma(VIVIENDAS_VIGO)
        suma_stgo = coef_suma(VIVIENDAS_SANTIAGO)
        print(f"   Coeficientes Vigo: {suma_vigo} | Santiago: {suma_stgo}")
        assert abs(suma_vigo - 1.0) < 0.01, f"Coef Vigo no suman 1: {suma_vigo}"
        assert abs(suma_stgo - 1.0) < 0.01, f"Coef Santiago no suman 1: {suma_stgo}"

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
        vigo = Comunidad(
            nombre='Residencial Vigo Centro',
            ubicacion='Vigo, Pontevedra',
            fecha_constitucion='2022-03-15',
            estado='activa',
            descripcion='Comunidad de autoconsumo colectivo (RD 244/2019) en el centro de Vigo. '
                        '15 viviendas, todas con paneles fotovoltaicos. Batería compartida de 20 kWh. '
                        'Coeficientes de reparto proporcionales a la potencia contratada.'
        )
        vigo_oid = save(vigo)

        santiago = Comunidad(
            nombre='Eco-Barrio Santiago',
            ubicacion='Santiago de Compostela, A Coruña',
            fecha_constitucion='2023-04-01',
            estado='activa',
            descripcion='Proyecto piloto de comunidad energética en 3 bloques residenciales '
                        'de Santiago de Compostela. 12 viviendas con batería de 30 kWh.'
        )
        santiago_oid = save(santiago)
        print(f"\n🏘️  2 comunidades creadas")

        # ------------------------------------------------------------------
        # Viviendas — Vigo
        # ------------------------------------------------------------------
        print(f"\n🏠 Creando viviendas Vigo ({len(VIVIENDAS_VIGO)})...")
        vigo_viv_oids = []
        for i, (ident, dir_, cups_suf, pot, coef, paneles, kwp, npan, orient) in enumerate(VIVIENDAS_VIGO):
            v = Vivienda(
                comunidad_oid=vigo_oid,
                identificador=ident,
                direccion_completa=f'{dir_}, 36202 Vigo',
                cups=f'ES003140500000{cups_suf}F',
                potencia_contratada_kw=pot,
                coeficiente_reparto=coef,
                fecha_alta='2022-04-01',
                tiene_paneles=paneles,
                potencia_pico_paneles_kwp=kwp,
                numero_paneles=npan,
                fecha_instalacion_paneles='2022-06-01' if paneles else None,
                orientacion_paneles=orient
            )
            oid = save(v)
            vigo_viv_oids.append(oid)

        # ------------------------------------------------------------------
        # Viviendas — Santiago
        # ------------------------------------------------------------------
        print(f"🏠 Creando viviendas Santiago ({len(VIVIENDAS_SANTIAGO)})...")
        stgo_viv_oids = []
        for i, (ident, dir_, cups_suf, pot, coef, paneles, kwp, npan, orient) in enumerate(VIVIENDAS_SANTIAGO):
            v = Vivienda(
                comunidad_oid=santiago_oid,
                identificador=ident,
                direccion_completa=f'{dir_}, 15705 Santiago de Compostela',
                cups=f'ES002130100000{cups_suf}F',
                potencia_contratada_kw=pot,
                coeficiente_reparto=coef,
                fecha_alta='2023-05-01',
                tiene_paneles=paneles,
                potencia_pico_paneles_kwp=kwp,
                numero_paneles=npan,
                fecha_instalacion_paneles='2023-07-01' if paneles else None,
                orientacion_paneles=orient
            )
            oid = save(v)
            stgo_viv_oids.append(oid)

        # ------------------------------------------------------------------
        # Baterías
        # ------------------------------------------------------------------
        save(Bateria(
            comunidad_oid=vigo_oid,
            capacidad_nominal_kwh=20.0,
            capacidad_util_actual_kwh=18.4,
            ciclos_acumulados=487,
            fecha_instalacion='2022-06-01',
            fabricante='BYD',
            modelo='Battery-Box Premium HVS 20.0',
            estado='operativa'
        ))
        save(Bateria(
            comunidad_oid=santiago_oid,
            capacidad_nominal_kwh=30.0,
            capacidad_util_actual_kwh=29.1,
            ciclos_acumulados=203,
            fecha_instalacion='2023-07-15',
            fabricante='Pylontech',
            modelo='H2 30kWh Stack',
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
                vivienda_oid=vigo_viv_oids[viv_idx],
                rol_en_vivienda=rol,
                fecha_incorporacion='2022-04-01'
            )
            save(a)
            n_accesos += 1

        # Accesos — Santiago
        for (viv_idx, usr_idx, rol) in ACCESOS_SANTIAGO:
            a = AccesoVivienda(
                usuario_oid=usuarios_oids[usr_idx],
                vivienda_oid=stgo_viv_oids[viv_idx],
                rol_en_vivienda=rol,
                fecha_incorporacion='2023-05-01'
            )
            save(a)
            n_accesos += 1

        print(f"   ✅ {n_accesos} accesos creados")

        # ------------------------------------------------------------------
        # Cierres mensuales
        # ------------------------------------------------------------------
        print(f"\n📊 Creando cierres mensuales ({len(MESES)} meses × {len(VIVIENDAS_VIGO)+len(VIVIENDAS_SANTIAGO)} viviendas)...")
        n_cierres = 0
        for idx, (viv_data, viv_oid) in enumerate(zip(VIVIENDAS_VIGO, vigo_viv_oids)):
            coef = viv_data[4]
            for mes in MESES:
                c = cierre_para_vivienda(viv_oid, mes, coef, idx)
                save(c)
                n_cierres += 1

        for idx, (viv_data, viv_oid) in enumerate(zip(VIVIENDAS_SANTIAGO, stgo_viv_oids)):
            coef = viv_data[4]
            tiene_paneles = viv_data[5]
            for mes in MESES:
                # Santiago: cierre con/sin paneles según configuración de cada vivienda
                if tiene_paneles:
                    c = cierre_para_vivienda(viv_oid, mes, coef, idx + 20)
                else:
                    from app.models.cierre import CierreMensual as CM
                    base = [310.0, 290.0, 275.0][MESES.index(mes)]
                    variacion = (((idx + 20) * 17) % 60) - 30
                    consumo = round(base + variacion, 1)
                    factura_base = round(consumo * 0.22, 2)
                    bateria = round(consumo * 0.13, 1)
                    kwh_red = max(0.0, consumo - bateria)
                    factura_real = round(kwh_red * 0.20 + bateria * 0.05, 2)
                    ahorro = round(max(0.0, factura_base - factura_real), 2)
                    c = CierreMensual(
                        vivienda_oid=viv_oid, mes=mes,
                        consumo_total_kwh=consumo, autoconsumo_directo_kwh=0.0,
                        energia_de_bateria_kwh=bateria, vertido_a_red_kwh=0.0,
                        ahorro_eur=ahorro,
                        factura_escenario_base_eur=factura_base,
                        factura_escenario_real_eur=factura_real if ahorro > 0 else factura_base,
                        porcentaje_ahorro_global=round(coef * 100, 1),
                        coeficiente_reparto_aplicado=coef
                    )
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

        # Para Carmen (admin Vigo)
        notif(1, 'general', '¡Bienvenida, administradora!',
              'Tu cuenta de administradora de Residencial Vigo Centro está activa.', leida=True)
        notif(1, 'cierre_disponible', 'Cierres de diciembre publicados',
              'Los cierres de 2024-12 ya están disponibles para todas las viviendas de tu comunidad.')
        notif(1, 'bateria_mantenimiento', 'Revisión anual de batería programada',
              'La batería BYD HVS 20.0 tiene programado su mantenimiento anual para enero de 2025.')

        # Para Marcos (admin Santiago)
        notif(2, 'general', '¡Bienvenido, administrador!',
              'Tu cuenta de administrador de Eco-Barrio Santiago está activa.', leida=True)
        notif(2, 'cierre_disponible', 'Cierres de diciembre publicados',
              'Los cierres de 2024-12 ya están disponibles para todas las viviendas.')
        notif(2, 'incidencia', 'Incidencia en inversor resuelta',
              'El fallo detectado en el inversor del Bloque C ha sido corregido por el técnico.')

        # Para usuarios normales (los primeros 6 vecinos)
        for usr_idx, nombre_corto in [(3,'Ana'), (4,'Luis'), (5,'María'), (6,'José'), (9,'Sara'), (10,'Pablo')]:
            notif(usr_idx, 'cierre_disponible', f'Cierre de diciembre disponible',
                  f'El cierre energético de 2024-12 de tu vivienda ya está disponible. '
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
║  Usuarios:   {len(usuarios_oids):3}   Viviendas: {len(vigo_viv_oids)+len(stgo_viv_oids):3}         ║
║  Accesos:    {n_accesos:3}   Cierres:   {n_cierres:3}         ║
╠══════════════════════════════════════════════════════╣
║  Credenciales:                                       ║
║  roberto@lealink.es     /  roberto1234  (superadmin) ║
║  carmen.vidal@vecinos.es / carmen1234   (admin Vigo) ║
║  marcos.iglesias@vecinos.es / marcos1234 (admin Stgo)║
║  ana.garcia@vecinos.es  /  ana1234      (vecina)     ║
╚══════════════════════════════════════════════════════╝
""")


if __name__ == '__main__':
    seed()
