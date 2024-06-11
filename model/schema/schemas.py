from marshmallow import Schema, fields, validate

from utils.functions import validate_date, validate_admin


class UserSchema(Schema):
    id = fields.Int(dump_only=True)
    name = fields.Str(required=True, validate=validate.Length(min=1))
    surname = fields.Str(required=True, validate=validate.Length(min=1))
    gender = fields.Str(required=True, validate=validate.OneOf(choices=['M', 'F']))
    date_of_birth = fields.Date(required=True, validate=validate_date)
    language = fields.Str(required=True, validate=validate.OneOf(choices=['EN', 'UA']))
    role = fields.Str(required=True, validate=validate.OneOf(choices=['USER', 'ADMIN']))
    id_registration = fields.Int(required=True, validate=validate.Range(min=1))


class RegistrationSchema(Schema):
    id = fields.Int(dump_only=True)
    email = fields.Email(required=True)
    encrypted_password = fields.Str(required=True, validate=validate.Length(min=8))
    registration_date_time = fields.DateTime(required=True, validate=validate_date)


class GesturesSchema(Schema):
    id = fields.Int(dump_only=True)
    name = fields.Str(required=True, validate=validate.Length(min=1))
    description = fields.Str(required=True, validate=validate.Length(min=5))
    language = fields.Str(required=True, validate=validate.OneOf(choices=['EN', 'UA']))


class CreationHistorySchema(Schema):
    id = fields.Int(dump_only=True)
    id_gesture = fields.Int(required=True, validate=validate.Range(min=1))
    creation_date_time = fields.DateTime(required=True, validate=validate_date)
    id_admin = fields.Int(required=True, validate=validate_admin)
