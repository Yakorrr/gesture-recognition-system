from flask.views import MethodView
from flask_jwt_extended import jwt_required
from flask_smorest import Blueprint, abort
from sqlalchemy.exc import IntegrityError, NoResultFound

from model import GestureSchema, GestureQuerySchema
from model import UserModel, GestureModel, CreationHistoryModel
from model.database import db

blp = Blueprint("gesture", __name__, description="Operations on gestures")


@blp.route("/gesture/<string:id>")
class Gesture(MethodView):
    @jwt_required()
    @blp.response(200, GestureSchema)
    def get(self, id_gesture):
        return GestureModel.query.get_or_404(id_gesture)

    @jwt_required()
    @blp.response(200, GestureSchema)
    def update(self, id_gesture):
        raise NotImplementedError("Not Implemented Now")

    @jwt_required()
    @blp.response(200, GestureSchema)
    def delete(self, id_gesture):
        raise NotImplementedError("Not Implemented Now")


@blp.route("/gesture")
class GestureList(MethodView):
    @jwt_required()
    @blp.response(200, GestureSchema(many=True))
    def get(self):
        return GestureModel.query.all()

    @jwt_required()
    @blp.arguments(GestureQuerySchema, location="query", as_kwargs=True)
    @blp.response(200, GestureSchema)
    def get(self, **kwargs):
        id_gesture = kwargs.get("id_gesture")
        gesture_name = kwargs.get("name")

        if not id_gesture and not gesture_name:
            abort(400, message="Bad request: Gesture ID or Gesture Name needed.")

        try:
            query = GestureModel.query.filter()

            if id_gesture:
                query = query.filter(GestureModel.id == id_gesture)
            if gesture_name:
                query = query.filter(GestureModel.name == gesture_name)
        except NoResultFound:
            abort(404, message="Bad request: Gesture not found")

        return query.all()

    @jwt_required()
    @blp.arguments(GestureSchema)
    @blp.response(200, GestureSchema)
    def post(self, gesture_data):
        try:
            admin = UserModel.query.filter(UserModel.id == gesture_data["id_admin"])

            if admin.role != "ADMIN":
                abort(
                    403,
                    message="You do not have enough rights to perform this operation."
                )
        except NoResultFound:
            abort(404, message="Bad request: Admin not found.")

        gesture = GestureModel(
            name=gesture_data["name"],
            description=gesture_data["description"],
            language=gesture_data["language"]
        )

        try:
            db.session.add(gesture)
            db.session.commit()
        except IntegrityError:
            abort(400, message="This gesture is already exist.")

        creation = CreationHistoryModel(
            id_gesture=gesture.id,
            id_admin=admin.id
        )

        try:
            db.session.add(gesture)
            db.session.add(creation)
            db.session.commit()
        except IntegrityError:
            abort(400, message="This gesture has already been added to the history.")

        return gesture
