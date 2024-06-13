from datetime import timedelta

from flask.views import MethodView
from flask_jwt_extended import create_access_token, jwt_required
from flask_smorest import Blueprint, abort
from passlib.handlers.pbkdf2 import pbkdf2_sha256
from sqlalchemy.exc import IntegrityError, NoResultFound

from model import UserModel, RegistrationModel
from model import UserSchema, UserRegistrationSchema, UserLoginSchema
from model.database import db

blp = Blueprint("user", __name__, description="Operations on user")


@blp.route("/user/<string:user_id>")
class User(MethodView):
    # Change registration_id to email and date of registration in future updates
    @jwt_required()
    @blp.response(200, UserSchema)
    def get(self, user_id):
        return UserModel.query.get_or_404(user_id)

    @jwt_required()
    @blp.response(200, UserSchema)
    def update(self, user_id):
        raise NotImplementedError("Not implemented now")

    @jwt_required()
    @blp.response(200, UserSchema)
    def delete(self, user_id):
        raise NotImplementedError("Not implemented now")


@blp.route("/users")
class UserList(MethodView):
    # Change registration_id to email and date of registration in future updates
    @jwt_required()
    @blp.response(200, UserSchema(many=True))
    def get(self):
        return UserModel.query.all()


@blp.route("/auth/register")
class AuthUser(MethodView):
    @blp.arguments(UserRegistrationSchema)
    @blp.response(200, UserRegistrationSchema)
    def post(self, registration_data):
        email = registration_data.get("email")
        password = registration_data.get("password")

        registration = RegistrationModel(
            email=email,
            encrypted_password=pbkdf2_sha256.hash(password)
        )

        try:
            db.session.add(registration)
            db.session.commit()
        except IntegrityError:
            abort(400, message="This email is already used")

        user = UserModel(
            name=registration_data.get("name"),
            surname=registration_data.get("surname"),
            gender=registration_data.get("gender"),
            date_of_birth=registration_data.get("date_of_birth"),
            language=registration_data.get("language"),
            role=registration_data.get("role"),
            id_registration=registration.id,
        )

        try:
            db.session.add(user)
            db.session.commit()
        except IntegrityError:
            abort(400, message="This user has already been created")

        return user


@blp.route("/auth/login")
class LoginUser(MethodView):
    @blp.arguments(UserLoginSchema)
    def post(self, login_data):
        login_email = login_data.get("email")
        password = login_data.get("password")

        try:
            registration = RegistrationModel.query.filter(
                RegistrationModel.email == login_email
            ).one()

            login_user = UserModel.query.filter(
                UserModel.id_registration == registration.id
            ).one()

            if login_user and pbkdf2_sha256.verify(
                    password, registration.encrypted_password):
                access_token = create_access_token(
                    identity=login_user.id,
                    expires_delta=timedelta(hours=4)
                )
            else:
                abort(403, message="Invalid credentials: Wrong password")

            db.session.add(login_user)
            db.session.commit()
        except NoResultFound:
            abort(404, message="Bad request: Email not found")
        except IntegrityError:
            abort(400, message="Bad request. Please try again later")

        return {"access_token": access_token}
