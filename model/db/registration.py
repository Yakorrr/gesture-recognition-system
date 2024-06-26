from sqlalchemy.sql import func

from model.database import db


class RegistrationModel(db.Model):
    __tablename__ = 'registrations'

    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), unique=True, nullable=False)
    encrypted_password = db.Column(db.String(255), nullable=False)

    registration_date_time = db.Column(
        db.TIMESTAMP,
        server_default=func.now()
    )

    user = db.relationship(
        "UserModel",
        back_populates="registration",
        lazy="dynamic"
    )
