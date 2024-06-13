from sqlalchemy import ForeignKey

from model.database import db


class UserModel(db.Model):
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(128), nullable=False)
    surname = db.Column(db.String(128), nullable=False)
    gender = db.Column(db.String, nullable=False)

    date_of_birth = db.Column(
        db.Date,
        nullable=False
    )

    language = db.Column(db.String, nullable=False)
    role = db.Column(db.String, nullable=False)

    id_registration = db.Column(
        db.Integer,
        ForeignKey("registrations.id", ondelete="CASCADE"),
        unique=True,
        nullable=False
    )

    registration = db.relationship(
        "RegistrationModel",
        foreign_keys=id_registration,
        back_populates="user"
    )

    creation_history = db.relationship(
        "CreationHistoryModel",
        back_populates="admin",
        lazy="dynamic"
    )
