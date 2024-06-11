from model.database import db


class GesturesModel(db.Model):
    __tablename__ = 'gestures'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String, unique=True, nullable=False)
    description = db.Column(db.String(128), nullable=False)
    language = db.Column(db.String, nullable=False)

    gesture = db.relationship(
        "CreationHistoryModel",
        back_populates="gestures",
        lazy="dynamic"
    )
