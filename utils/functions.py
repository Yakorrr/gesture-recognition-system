import csv
import datetime

from marshmallow import ValidationError, validates

from model.database import db
from model.db.user import UserModel


def find_symbol_classifier_label(symbol):
    with (open('model/keypoint_classifier/keypoint_classifier_label.csv',
               'r', newline='', encoding='utf-8') as file):
        reader = csv.reader(file)
        print("Key pressed:", symbol, end='\n')

        # Iterate through each row in the CSV file
        for i, row in enumerate(reader):
            joined_row = ''.join(row)

            # Convert the row to a string and check if it contains the symbol
            if (symbol.upper() + ' ') in joined_row and \
                    ('EN' in joined_row or 'UA' in joined_row):
                return i

    return None


def process_numbers(key_code):
    return key_code - ord('0')


def process_letters(key_code: int):
    if 224 <= key_code <= 255:
        key_code += ord('а') - ord('а'.encode('cp1251'))

    row_number = find_symbol_classifier_label(chr(key_code))
    print("Row number:", row_number)

    if row_number is not None:
        return row_number

    return None


def select_mode(key, mode):
    # Eng: 65-90 (upper), 97-122 (lower)
    # Ukr: 192-223 (upper), 224-255 (lower)
    ukrainian_range = range(ord('а'.encode('cp1251')), ord('я'.encode('cp1251')) + 1)
    english_range = range(ord('a'), ord('z') + 1)
    numbers_range = range(ord('0'), ord('5') + 1)

    classifier_table_row_number = -1

    if key in numbers_range:  # Numbers 0 ~ 9
        classifier_table_row_number = process_numbers(key)
        print(classifier_table_row_number)
    if (key in english_range or  # English letters (both upper and lower)
            key in ukrainian_range):  # Ukrainian letters (both upper and lower)
        classifier_table_row_number = process_letters(key)

    if key == ord('8'):  # Key 8 - Normal Mode
        mode = 0
    if key == ord('9'):  # Key 9 - Save Key Points Mode
        mode = 1

    return classifier_table_row_number, mode


def logging_csv(number, mode, landmark_list):
    if mode == 0:
        pass

    if mode == 1 and (0 <= number <= 63):
        csv_path = 'model/keypoint_classifier/keypoint.csv'

        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])


def validate_date(date):
    if date <= datetime.date(1900, 1, 1):
        raise ValidationError("Date must be greater than or equal to January 1, 1900.")


@validates('id_admin')
def validate_admin(value):
    # Query the database to check if the provided id_admin is an admin
    admin_user = db.session.query(UserModel).filter_by(id=value, role='ADMIN').first()

    if not admin_user:
        raise ValidationError("id_admin must be the ID of a user with the role 'ADMIN'.")
