from flask.views import MethodView
from flask_jwt_extended import create_access_token, jwt_required
from flask_smorest import Blueprint, abort
from passlib.handlers.pbkdf2 import pbkdf2_sha256
from sqlalchemy.exc import IntegrityError, NoResultFound

from model import UserModel, RegistrationModel
from model import UserSchema, RegistrationSchema, RegistrationQuerySchema
from model.database import db

blp = Blueprint("registration", __name__, description="Operations on registration")


@blp.route("/registration/<string:id>")
class Registration(MethodView):
    @jwt_required()
    @blp.response(200, RegistrationSchema)
    def get(self, id_reg):
        return RegistrationModel.query.get_or_404(id_reg)

    @jwt_required()
    @blp.response(200, RegistrationSchema)
    def update(self, id_reg):
        raise NotImplementedError("Not Implemented Now")

    @jwt_required()
    @blp.response(200, RegistrationSchema)
    def delete(self, id_reg):
        raise NotImplementedError("Not Implemented Now")


@blp.route("/registration")
class RegistrationList(MethodView):
    @jwt_required()
    @blp.response(200, RegistrationSchema(many=True))
    def get(self):
        return RegistrationModel.query.all()

    @jwt_required()
    @blp.arguments(RegistrationQuerySchema, location="query", as_kwargs=True)
    @blp.response(200, RegistrationSchema)
    def get(self, **kwargs):
        id_reg = kwargs.get("id_registration")
        email = kwargs.get("email")

        if not id_reg and not email:
            abort(400, message="Bad request: Registration ID or Email needed.")

        try:
            query = RegistrationModel.query.filter()

            if id_reg:
                query = query.filter(RegistrationModel.id == id_reg)
            if email:
                query = query.filter(RegistrationModel.email == email)
        except NoResultFound:
            abort(404, message="Bad request: Registration not found")

        return query.all()
