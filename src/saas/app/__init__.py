"""Factory de la aplicación LeaLink."""
from flask import Flask, render_template
from flask_login import LoginManager, current_user
from flask_wtf import CSRFProtect

from app.extensions import db

login_manager = LoginManager()
csrf = CSRFProtect()


def create_app(config_name=None):
    """Crea y configura la aplicación Flask."""
    app = Flask(__name__)

    from config import get_config
    app.config.from_object(get_config(config_name))

    # Inicializar la base de datos (SQLAlchemy / PostgreSQL)
    db.init_app(app)

    # Extensiones
    login_manager.init_app(app)
    csrf.init_app(app)
    login_manager.login_view = 'auth.login'
    login_manager.login_message = 'Inicia sesión para acceder a esta página.'
    login_manager.login_message_category = 'warning'

    # Funciones disponibles globalmente en todos los templates
    from app.helpers import oid_to_safe
    from markupsafe import Markup
    from flask_wtf.csrf import generate_csrf

    app.jinja_env.globals['oid_to_safe'] = oid_to_safe

    def csrf_token_hidden():
        return Markup(f'<input type="hidden" name="csrf_token" value="{generate_csrf()}">')

    app.jinja_env.globals['csrf_token_hidden'] = csrf_token_hidden

    # Loader de usuario para Flask-Login
    @login_manager.user_loader
    def load_user(user_id):
        from app.models.usuario import Usuario
        try:
            return db.session.get(Usuario, int(user_id))
        except Exception:
            return None

    # Context processor: globals disponibles en todos los templates
    @app.context_processor
    def inject_globals():
        def contador_no_leidas():
            if not current_user.is_authenticated:
                return 0
            from app.models.notificacion import Notificacion
            return Notificacion.query.filter_by(
                usuario_oid=current_user.id, leida=False
            ).count()

        def mi_comunidad_nav():
            """Retorna lista de {safe_oid, nombre} de las comunidades del usuario, o []."""
            if not current_user.is_authenticated or current_user.es_superadmin:
                return []
            from app.models.acceso import AccesoVivienda
            from app.models.vivienda import Vivienda
            from app.models.comunidad import Comunidad
            from app.helpers import oid_to_safe
            try:
                accesos = AccesoVivienda.query.filter_by(
                    usuario_oid=current_user.id
                ).all()
                vistas = {}
                for a in accesos:
                    viv = db.session.get(Vivienda, a.vivienda_oid)
                    if not viv or viv.comunidad_oid in vistas:
                        continue
                    com = db.session.get(Comunidad, viv.comunidad_oid)
                    if com:
                        vistas[viv.comunidad_oid] = {
                            'safe_oid': oid_to_safe(com.__oid__),
                            'nombre': com.nombre,
                        }
                return list(vistas.values())
            except Exception:
                return []

        return {
            'contador_no_leidas': contador_no_leidas,
            'mi_comunidad_nav': mi_comunidad_nav,
        }

    # Registro de blueprints
    from app.modules.auth import auth_bp
    from app.modules.main import main_bp
    from app.modules.comunidades import comunidades_bp
    from app.modules.viviendas import viviendas_bp
    from app.modules.usuarios import usuarios_bp
    from app.modules.accesos import accesos_bp
    from app.modules.baterias import baterias_bp
    from app.modules.cierres import cierres_bp
    from app.modules.notificaciones import notificaciones_bp
    from app.modules.incidencias import incidencias_bp
    from app.modules.edge import edge_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(main_bp)
    app.register_blueprint(comunidades_bp, url_prefix='/comunidades')
    app.register_blueprint(viviendas_bp, url_prefix='/viviendas')
    app.register_blueprint(usuarios_bp, url_prefix='/usuarios')
    app.register_blueprint(accesos_bp, url_prefix='/accesos')
    app.register_blueprint(baterias_bp, url_prefix='/baterias')
    app.register_blueprint(cierres_bp, url_prefix='/cierres')
    app.register_blueprint(notificaciones_bp, url_prefix='/notificaciones')
    app.register_blueprint(incidencias_bp)
    app.register_blueprint(edge_bp, url_prefix='/api/edge')
    csrf.exempt(edge_bp)  # API machine-to-machine: auth por Bearer token, no CSRF

    # Healthcheck para Docker Compose
    from flask import jsonify as _jsonify

    @app.route('/health')
    def health():
        return _jsonify({'status': 'ok'}), 200

    # Manejadores de error personalizados
    @app.errorhandler(401)
    def no_autorizado(e):
        return render_template('errors/401.html'), 401

    @app.errorhandler(403)
    def prohibido(e):
        return render_template('errors/403.html'), 403

    @app.errorhandler(404)
    def no_encontrado(e):
        return render_template('errors/404.html'), 404

    @app.errorhandler(500)
    def error_interno(e):
        return render_template('errors/500.html'), 500

    # Crear el esquema si no existe (suficiente para el alcance del TFG).
    # Importar los modelos garantiza que todas las tablas estén registradas.
    import app.models as _models  # noqa: F401  (registra los modelos en el metadata)
    with app.app_context():
        db.create_all()

    return app
