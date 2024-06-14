import argparse
import copy
import itertools
from collections import deque

import cv2 as cv
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from flask_main import create_app
from model import KeyPointClassifier
from utils import CvFpsCalc
from utils.functions import (language_buttons_callback,
                             select_mode, logging_csv,
                             get_gesture_name)


class App:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.args = self.get_args()

        self.cap_device = self.args.device
        self.cap_width = self.args.width
        self.cap_height = self.args.height

        self.mode = 0  # Default Mode is Translation
        self.language = 'EN'  # Default Language is English

        self.button_params = None

    def get_args(self):
        self.parser.add_argument("--device", type=int, default=0)
        self.parser.add_argument("--width", help='cap width', type=int, default=1280)
        self.parser.add_argument("--height", help='cap height', type=int, default=800)

        self.parser.add_argument('--use_static_image_mode', action='store_true')
        self.parser.add_argument("--min_detection_confidence",
                                 help='min_detection_confidence',
                                 type=float,
                                 default=0.7)
        self.parser.add_argument("--min_tracking_confidence",
                                 help='min_tracking_confidence',
                                 type=int,
                                 default=0.5)

        args = self.parser.parse_args()

        return args

    def main(self):
        create_app()

        # Argument parsing #################################################################
        use_static_image_mode = self.args.use_static_image_mode
        min_detection_confidence = self.args.min_detection_confidence
        min_tracking_confidence = self.args.min_tracking_confidence

        use_brect = True

        # Camera preparation ###############################################################
        cap = cv.VideoCapture(self.cap_device)
        cap.set(cv.CAP_PROP_FRAME_WIDTH, self.cap_width)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.cap_height)

        # Model load #############################################################
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=use_static_image_mode,
            max_num_hands=2,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        keypoint_classifier = KeyPointClassifier()

        # FPS Measurement ########################################################
        cv_fps_calc = CvFpsCalc(buffer_len=10)

        # Coordinate history #################################################################
        history_length = 16
        point_history = deque(maxlen=history_length)

        while True:
            fps = cv_fps_calc.get()

            # Process Key (ESC: end) #################################################
            key = cv.waitKey(10)

            if key == 27:  # ESC
                break

            number, self.mode = select_mode(key, self.mode, self.language)

            # Camera capture #####################################################
            ret, image = cap.read()

            if not ret:
                break

            image = cv.flip(image, 1)  # Mirror display
            debug_image = copy.deepcopy(image)

            # Detection implementation #############################################################
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

            image.flags.writeable = False
            results = hands.process(image)
            image.flags.writeable = True

            #  ####################################################################
            if results.multi_hand_landmarks is not None:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                      results.multi_handedness):
                    # Bounding box calculation
                    brect = self.calc_bounding_rect(debug_image, hand_landmarks)

                    # Landmark calculation
                    landmark_list = self.calc_landmark_list(debug_image, hand_landmarks)

                    # Conversion to relative coordinates / normalized coordinates
                    pre_processed_landmark_list = self.pre_process_landmark(
                        landmark_list
                    )

                    # Write to the dataset file
                    logging_csv(number, self.mode, pre_processed_landmark_list)

                    # Hand sign classification
                    hand_sign_id = keypoint_classifier(pre_processed_landmark_list)

                    keypoint_label = get_gesture_name(
                        hand_sign_id,
                        self.language
                    ) or "Undefined"

                    # Drawing part
                    debug_image = self.draw_bounding_rect(use_brect, debug_image, brect)
                    debug_image = self.draw_landmarks(debug_image, landmark_list)
                    debug_image = self.draw_info_text(
                        debug_image,
                        brect,
                        handedness,
                        keypoint_label,
                    )
            else:
                point_history.append([0, 0])

            debug_image = self.draw_point_history(debug_image, point_history)
            debug_image = self.draw_language_buttons(debug_image)
            debug_image = self.draw_info(debug_image, fps, self.mode, number)

            # Screen reflection #############################################################
            cv.imshow('Hand Gesture Recognition', debug_image)
            cv.setMouseCallback(
                'Hand Gesture Recognition',
                language_buttons_callback,
                {
                    'button_params': self.button_params,
                    'app_instance': self
                }
            )

        cap.release()
        cv.destroyAllWindows()

    def draw_language_buttons(self, image):
        # 'debug_image' is the current frame from webcam
        height, width = image.shape[:2]

        # Display language buttons
        button_width = 60
        button_height = 30
        top_margin = 20
        left_margin = width - button_width - 20

        # English Button
        button_en_x1, button_en_y1 = left_margin, top_margin
        button_en_x2, button_en_y2 = left_margin + button_width, top_margin + button_height

        cv.rectangle(image, (left_margin, top_margin),
                     (left_margin + button_width, top_margin + button_height),
                     (255, 204, 0), -1)
        cv.putText(image, "EN",
                   (left_margin + int(button_width / 2) - 10,
                    top_margin + int(button_height / 2) + 5),
                   cv.FONT_HERSHEY_SIMPLEX, 0.4,
                   (255, 255, 255), 1, cv.LINE_AA)

        # Ukrainian Button
        button_ua_x1, button_ua_y1 = left_margin, top_margin + button_height + 5
        button_ua_x2, button_ua_y2 = left_margin + button_width, top_margin + 2 * button_height + 5

        cv.rectangle(image, (left_margin, top_margin + button_height + 5),
                     (left_margin + button_width, top_margin + 2 * button_height + 5),
                     (46, 139, 87), -1)
        cv.putText(image, "UA",
                   (left_margin + int(button_width / 2) - 10,
                    top_margin + button_height + int(button_height / 2) + 10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.4,
                   (255, 255, 255), 1, cv.LINE_AA)

        # Return the coordinates of the buttons
        self.button_params = {
            'button_en': {'x1': button_en_x1, 'y1': button_en_y1, 'x2': button_en_x2, 'y2': button_en_y2},
            'button_ua': {'x1': button_ua_x1, 'y1': button_ua_y1, 'x2': button_ua_x2, 'y2': button_ua_y2}
        }

        return image

    @staticmethod
    def calc_bounding_rect(image, landmarks):
        image_width, image_height = image.shape[1], image.shape[0]

        landmark_array = np.empty((0, 2), int)

        for _, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)

            landmark_point = [np.array((landmark_x, landmark_y))]

            landmark_array = np.append(landmark_array, landmark_point, axis=0)

        x, y, w, h = cv.boundingRect(landmark_array)

        return [x, y, x + w, y + h]

    @staticmethod
    def calc_landmark_list(image, landmarks):
        image_width, image_height = image.shape[1], image.shape[0]

        landmark_point = []

        # Keypoint
        for _, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)

            landmark_point.append([landmark_x, landmark_y])

        return landmark_point

    @staticmethod
    def pre_process_landmark(landmark_list):
        temp_landmark_list = copy.deepcopy(landmark_list)

        # Convert to relative coordinates
        base_x, base_y = 0, 0

        for index, landmark_point in enumerate(temp_landmark_list):
            if index == 0:
                base_x, base_y = landmark_point[0], landmark_point[1]

            temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
            temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

        # Convert to a one-dimensional list
        temp_landmark_list = list(
            itertools.chain.from_iterable(temp_landmark_list))

        # Normalization
        max_value = max(list(map(abs, temp_landmark_list)))

        def normalize_(n):
            return n / max_value

        temp_landmark_list = list(map(normalize_, temp_landmark_list))

        return temp_landmark_list

    @staticmethod
    def draw_landmarks(image, landmark_point):
        if len(landmark_point) > 0:
            # Thumb
            cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                    (255, 255, 255), 2)

            # Index finger
            cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                    (255, 255, 255), 2)

            # Middle finger
            cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                    (255, 255, 255), 2)

            # Ring finger
            cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                    (255, 255, 255), 2)

            # Little finger
            cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                    (255, 255, 255), 2)

            # Palm
            cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                    (255, 255, 255), 2)

        # Key Points
        for index, landmark in enumerate(landmark_point):
            if index == 0:  # Wrist 1 (WRIST)
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 1:  # Thumb: Base (THUMB_CMP)
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 2:  # Thumb: 1st joint (THUMB_MCP)
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 3:  # Thumb: 2nd joint (THUMP_IP)
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 4:  # Thumb: fingertip (THUMB_TIP)
                cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
            if index == 5:  # Index finger: base (INDEX_FINGER_MCP)
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 6:  # Index finger: 1st joint (INDEX_FINGER_PIP)
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 7:  # Index finger: 2nd joint (INDEX_FINGER_DIP)
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 8:  # Index finger: fingertip (INDEX_FINGER_TIP)
                cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
            if index == 9:  # Middle finger: base (MIDDLE_FINGER_MCP)
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 10:  # Middle finger: 1st joint (MIDDLE_FINGER_PIP)
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 11:  # Middle finger: 2nd joint (MIDDLE_FINGER_DIP)
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 12:  # Middle finger: fingertip (MIDDLE_FINGER_TIP)
                cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
            if index == 13:  # Ring finger: base (RING_FINGER_MCP)
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 14:  # Ring finger: 1st joint (RING_FINGER_PIP)
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 15:  # Ring finger: 2nd joint (RING_FINGER_DIP)
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 16:  # Ring finger: fingertip (RING_FINGER_TIP)
                cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
            if index == 17:  # Little finger: base (PINKY_MCP)
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 18:  # Little finger: 1st joint (PINKY_PIP)
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 19:  # Little finger: 2nd joint (PINKY_DIP)
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 20:  # Little finger: fingertip (PINKY_TIP)
                cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

        return image

    @staticmethod
    def draw_bounding_rect(use_brect, image, brect):
        if use_brect:
            # Outer rectangle
            cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                         (0, 0, 0), 1)

        return image

    @staticmethod
    def draw_info_text(image, brect, handedness, hand_sign_text):
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                     (0, 0, 0), -1)

        # Convert the OpenCV image to a PIL Image
        cv_rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        pil_image = Image.fromarray(cv_rgb_image)

        draw = ImageDraw.Draw(pil_image)

        # Define text and position
        info_text = handedness.classification[0].label[0:]

        if hand_sign_text != "":
            info_text += ': ' + hand_sign_text

        position = (brect[0] + 5, brect[1] - 24)

        # Define the font (make sure the path to the font is correct, and it supports Cyrillic characters)
        font = ImageFont.truetype("utils/fonts/Montserrat-Regular.ttf", 20)  # Adjust font size if needed

        # Draw text
        draw.text(position, info_text, font=font, fill=(255, 255, 255))

        # Convert back to OpenCV image
        image = cv.cvtColor(np.array(pil_image), cv.COLOR_RGB2BGR)

        return image

    @staticmethod
    def draw_point_history(image, point_history):
        for index, point in enumerate(point_history):
            if point[0] != 0 and point[1] != 0:
                cv.circle(image, (point[0], point[1]), 1 + int(index / 2),
                          (152, 251, 152), 2)

        return image

    def draw_info(self, image, fps, mode, number):
        cv.putText(image, "FPS: " + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
                   1.0, (255, 255, 255), 2, cv.LINE_AA)

        mode_string = [
            'Translation',
            'Logging Key Point',
        ]

        if 0 <= mode <= 1:
            cv.putText(image, "Mode: " + mode_string[mode], (10, 90),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                       cv.LINE_AA)

            if 0 <= number <= 255:
                cv.putText(image, "Key pressed: " + str(number), (10, 120),
                           cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                           cv.LINE_AA)

        # Display current language
        cv.putText(image, f"Current Language: {self.language}",
                   (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.6,
                   (255, 255, 255), 1, cv.LINE_AA)

        return image


if __name__ == '__main__':
    app = App()
    app.main()
