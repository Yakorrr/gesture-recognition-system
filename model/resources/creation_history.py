from flask.views import MethodView
from flask_jwt_extended import jwt_required
from flask_smorest import Blueprint, abort
from sqlalchemy import and_
from sqlalchemy.exc import IntegrityError

from model import CreationHistoryModel
from model import CreationHistorySchema, CreationHistoryQuerySchema

blp = Blueprint("creation_history", __name__, description="Gesture Creation History")


@blp.route("/creation-history/<string:id>")
class CreationHistory(MethodView):
    @jwt_required()
    @blp.response(200, CreationHistorySchema)
    def get(self, gesture_id):
        return CreationHistoryModel.query.get_or_404(gesture_id)

    @jwt_required()
    @blp.response(200, CreationHistorySchema)
    def delete(self, gesture_id):
        raise NotImplementedError("Not Implemented Now")


@blp.route("/creation-history")
class CreationHistoryList(MethodView):
    @jwt_required()
    @blp.response(200, CreationHistorySchema(many=True))
    def get(self):
        return CreationHistoryModel.query.all()


@blp.route("/creation")
class Creation(MethodView):
    # Return a json that contains admin name and gesture name instead of ID's in future updates
    @jwt_required()
    @blp.arguments(CreationHistoryQuerySchema, location="query", as_kwargs=True)
    @blp.response(200, CreationHistoryQuerySchema(many=True))
    def get(self, **kwargs):
        id_gesture = kwargs.get("id_gesture")
        id_admin = kwargs.get("id_admin")

        if not id_gesture and not id_admin:
            abort(400, message="Bad request: Gesture ID or Admin ID needed")

        try:

            query = CreationHistoryModel.query
            conditions = []

            if id_gesture:
                conditions.append(CreationHistoryModel.id_gesture == id_gesture)
            if id_admin:
                conditions.append(CreationHistoryModel.id_admin == id_admin)

            if conditions:
                query = query.filter(and_(*conditions))

            history = query.all()

            if not history:
                abort(404, message="Bad request: No data found")

        except IntegrityError as e:
            abort(400, message=str(e))

        return history
