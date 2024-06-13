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


class UserRegistrationSchema(Schema):
    name = fields.Str(required=True, validate=validate.Length(min=1))
    surname = fields.Str(required=True, validate=validate.Length(min=1))
    gender = fields.Str(required=True, validate=validate.OneOf(choices=['M', 'F']))
    date_of_birth = fields.Date(required=True, validate=validate_date)
    language = fields.Str(required=True, validate=validate.OneOf(choices=['EN', 'UA']))
    role = fields.Str(required=True, validate=validate.OneOf(choices=['USER', 'ADMIN']))
    email = fields.Email(required=True)
    password = fields.Str(required=True, validate=validate.Length(min=8))


class UserLoginSchema(Schema):
    email = fields.Email(required=True)
    password = fields.Str(required=True)


class RegistrationSchema(Schema):
    id = fields.Int(dump_only=True)
    email = fields.Email(required=True)
    encrypted_password = fields.Str(required=True, validate=validate.Length(min=8))
    registration_date_time = fields.DateTime(required=True, validate=validate_date)


class RegistrationQuerySchema(Schema):
    reg_id = fields.Int(validate=validate.Range(min=1))
    email = fields.Str(validate=validate.Email())


class GestureSchema(Schema):
    id = fields.Int(dump_only=True)
    name = fields.Str(required=True, validate=validate.Length(min=1))
    description = fields.Str(required=True, validate=validate.Length(min=5))
    language = fields.Str(required=True, validate=validate.OneOf(choices=['EN', 'UA']))


class GestureQuerySchema(Schema):
    id_gesture = fields.Int(validate=validate.Range(min=1))
    name = fields.Str(validate=validate.Length(min=1))


class GestureInsertionSchema(Schema):
    name = fields.Str(required=True, validate=validate.Length(min=1))
    description = fields.Str(required=True, validate=validate.Length(min=5))
    language = fields.Str(required=True, validate=validate.OneOf(choices=['EN', 'UA']))
    id_admin = fields.Int(required=True, validate=validate.Range(min=1))


class CreationHistorySchema(Schema):
    id = fields.Int(dump_only=True)
    id_gesture = fields.Int(required=True, validate=validate.Range(min=1))
    creation_date_time = fields.DateTime(required=True, validate=validate_date)
    id_admin = fields.Int(required=True, validate=validate_admin)


class CreationHistoryQuerySchema(Schema):
    id_gesture = fields.Int(validate=validate.Range(min=1))
    id_admin = fields.Int(validate=validate.Range(min=1))
    creation_date_time = fields.DateTime(validate=validate_date)
