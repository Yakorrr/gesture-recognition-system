from sqlalchemy import ForeignKey
from sqlalchemy.sql import func

from model.database import db


class CreationHistoryModel(db.Model):
    __tablename__ = 'creation_history'

    id = db.Column(db.Integer, primary_key=True)

    id_gesture = db.Column(
        db.Integer,
        ForeignKey("gestures.id", ondelete="CASCADE"),
        unique=True,
        nullable=False
    )

    creation_date_time = db.Column(
        db.TIMESTAMP,
        server_default=func.now(),
        nullable=False
    )

    id_admin = db.Column(
        db.Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False
    )

    gesture = db.relationship(
        "GestureModel",
        foreign_keys=id_gesture,
        back_populates="creation_history"
    )

    admin = db.relationship(
        "UserModel",
        foreign_keys=id_admin,
        back_populates="creation_history"
    )
