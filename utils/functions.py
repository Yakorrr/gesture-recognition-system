import csv
import datetime

import cv2 as cv
from marshmallow import ValidationError, validates

from model.database import db
from model.db.user import UserModel


def language_buttons_callback(event, x, y, _, params):
    app_instance = params['app_instance']
    button_en, button_ua = (params['button_params']['button_en'],
                            params['button_params']['button_ua'])

    if event == cv.EVENT_LBUTTONDOWN:
        if (button_en['x1'] <= x <= button_en['x2'] and
                button_en['y1'] <= y <= button_en['y2']):  # Coordinates for the English button
            app_instance.language = 'EN'
        elif (button_ua['x1'] <= x <= button_ua['x2'] and
              button_ua['y1'] <= y <= button_ua['y2']):  # Coordinates for the Ukrainian button
            app_instance.language = 'UA'


def get_gesture_name(class_number, language):
    with (open('model/keypoint_classifier/keypoint_classifier_label.csv',
               'r', newline='', encoding='utf-8-sig') as file):
        class_name = list(csv.reader(file))[class_number][0]

        if (('(' not in class_name and ')' not in class_name) or
                language in class_name):
            return class_name

    return None


def find_symbol_classifier_label(symbol, language):
    with (open('model/keypoint_classifier/keypoint_classifier_label.csv',
               'r', newline='', encoding='utf-8') as file):
        # Iterate through each row in the CSV file
        for i, row in enumerate(csv.reader(file)):
            joined_row = ''.join(row)

            # Convert the row to a string and check if it contains the symbol
            if (symbol.upper() + ' ') in joined_row and language in joined_row:
                return i

    return None


def process_numbers(key_code):
    return key_code - ord('0')


def process_letters(key_code: int, language):
    if 224 <= key_code <= 255:
        key_code += ord('а') - ord('а'.encode('cp1251'))

    return find_symbol_classifier_label(chr(key_code), language) or -1


def select_mode(key, mode, language):
    # Eng: 65-90 (upper), 97-122 (lower)
    # Ukr: 192-223 (upper), 224-255 (lower)
    ukrainian_range = range(ord('а'.encode('cp1251')), ord('я'.encode('cp1251')) + 1)
    english_range = range(ord('a'), ord('z') + 1)
    numbers_range = range(ord('0'), ord('5') + 1)

    classifier_table_row_number = -1

    if key in numbers_range:  # Numbers 0 ~ 9
        classifier_table_row_number = process_numbers(key)
    if (key in english_range or  # English letters (both upper and lower)
            key in ukrainian_range):  # Ukrainian letters (both upper and lower)
        classifier_table_row_number = process_letters(key, language)

    if key == ord('8'):  # Key 8 - Normal Mode
        mode = 0
    if key == ord('9'):  # Key 9 - Save Key Points Mode
        mode = 1

    return classifier_table_row_number, mode


def logging_csv(number, mode, landmark_list):
    if mode == 0 or number < 0:
        pass

    if mode == 1 and (0 <= number <= 63):
        csv_path = 'model/keypoint_classifier/keypoint.csv'

        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])


def validate_date(date):
    if date <= datetime.date(1900, 1, 1):
        raise ValidationError("Date must be greater than or equal to January 1, 1900")


@validates('id_admin')
def validate_admin(value):
    # Query the database to check if the provided id_admin is an admin
    admin_user = (db.session.query(UserModel)
                  .filter_by(id=value, role='ADMIN').first())

    if not admin_user:
        raise ValidationError("id_admin must be the ID of a user with the role 'ADMIN'")
