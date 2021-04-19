"""Initialize Flask app."""
from flask import Flask
#from flask_assets import Environment
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

from flask_login import LoginManager
db = SQLAlchemy()
migrate = Migrate()
login_manager = LoginManager()
# from app.router import Router
from app.model import Model
from werkzeug.middleware.proxy_fix import ProxyFix


def init_app():
    """Construct core Flask application with embedded Dash app."""
    web = Flask(__name__, instance_relative_config=False)
    web.config.from_object('config.Config')

    #Initializing plugins
    db.init_app(web)
    #login_manager.init_app(web)
    model = Model(web)

    with web.app_context():
        # Import parts of our core Flask app
        from . import route

        # Import Dash application
        from .plotlydash.dashboard import init_dashboard
        web = init_dashboard(web)

        #Create SQL tables
        migrate.init_app(web, db)

        return web