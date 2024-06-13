from flask.views import MethodView
from flask_jwt_extended import jwt_required
from flask_smorest import Blueprint, abort
from sqlalchemy import and_
from sqlalchemy.exc import NoResultFound

from model import RegistrationModel
from model import RegistrationSchema, RegistrationQuerySchema

blp = Blueprint("registration", __name__, description="Operations on registration")


@blp.route("/registration/<string:id>")
class Registration(MethodView):
    @jwt_required()
    @blp.response(200, RegistrationSchema)
    def get(self, reg_id):
        return RegistrationModel.query.get_or_404(reg_id)

    @jwt_required()
    @blp.response(200, RegistrationSchema)
    def update(self, reg_id):
        raise NotImplementedError("Not Implemented Now")

    @jwt_required()
    @blp.response(200, RegistrationSchema)
    def delete(self, reg_id):
        raise NotImplementedError("Not Implemented Now")


@blp.route("/registrations")
class RegistrationList(MethodView):
    @jwt_required()
    @blp.response(200, RegistrationSchema(many=True))
    def get(self):
        return RegistrationModel.query.all()


@blp.route("/registration")
class RegistrationView(MethodView):
    @jwt_required()
    @blp.arguments(RegistrationQuerySchema, location="query", as_kwargs=True)
    @blp.response(200, RegistrationSchema)
    def get(self, **kwargs):
        reg_id = kwargs.get("reg_id")
        email = kwargs.get("email")

        if not reg_id and not email:
            abort(400, message="Bad request: Registration ID or Email needed")

        registration = None

        try:
            query = RegistrationModel.query
            conditions = []

            if reg_id:
                conditions.append(RegistrationModel.id == reg_id)
            if email:
                conditions.append(RegistrationModel.email == email)

            if conditions:
                registration = query.filter(and_(*conditions)).one()
        except NoResultFound:
            abort(404, message="Bad request: Registration not found")

        return registration
