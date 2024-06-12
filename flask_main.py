import os

from flask import Flask, jsonify
from flask_jwt_extended import JWTManager
from flask_login import LoginManager
from flask_smorest import Api, Blueprint

from model.database import db
from model.resources.creation_history import blp as CreationHistoryBlueprint
from model.resources.gesture import blp as GestureBlueprint
from model.resources.registration import blp as RegistrationBlueprint
from model.resources.user import blp as UserBlueprint

defaultPage = Blueprint("index", __name__, description="Default page")


def create_app():
    login_manager = LoginManager()

    app = Flask(__name__)
    app.config["PROPAGATE_EXCEPTIONS"] = True
    app.config["API_TITLE"] = "Gesture Recognition System"
    app.config["API_VERSION"] = "v1.0"
    app.config["OPENAPI_VERSION"] = "3.0.3"
    app.config["OPENAPI_URL_PREFIX"] = "/"
    app.config["OPENAPI_SWAGGER_UI_PATH"] = "/swagger-ui"
    app.config["OPENAPI_SWAGGER_UI_URL"] = "https://cdn.jsdelivr.net/npm/swagger-ui-dist/"
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///data.db"
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    app.config["JWT_SECRET_KEY"] = os.getenv("JWT_SECRET_KEY")

    login_manager.init_app(app)
    db.init_app(app)

    with app.app_context():
        db.create_all()

    api = Api(app)

    jwt = JWTManager(app)

    @jwt.expired_token_loader
    def expired_token_callback():
        return (
            jsonify({
                "message": "The token has expired.",
                "error": "token_expired"
            }),
            401,
        )

    @jwt.invalid_token_loader
    def invalid_token_callback():
        return (
            jsonify(
                {
                    "message": "Signature verification failed.",
                    "error": "invalid_token"
                }
            ),
            401,
        )

    @jwt.unauthorized_loader
    def missing_token_callback():
        return (
            jsonify(
                {
                    "description": "Request does not contain an access token.",
                    "error": "authorization_required",
                }
            ),
            401,
        )

    api.register_blueprint(defaultPage)
    api.register_blueprint(UserBlueprint)
    api.register_blueprint(RegistrationBlueprint)
    api.register_blueprint(GestureBlueprint)
    api.register_blueprint(CreationHistoryBlueprint)

    return app


@defaultPage.route("/")
def index():
    return jsonify("Welcome to the Gesture Recognition System!")
