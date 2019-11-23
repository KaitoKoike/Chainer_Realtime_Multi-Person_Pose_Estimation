import pickle
import cv2
import numpy as np


class GestureRecognizer(object):
    def __init__(self, model_path=None):
        print('Loading GestureModel...')
        print(model_path)
        self.model = pickle.load(open(model_path, "rb"))

    def __call__(self, hand_keypoint):
        hand_keypoint = self.preprocessing(hand_keypoint)
        print(hand_keypoint)
        gesture = self.model.predict([hand_keypoint])
        print(gesture)
        return gesture

    def preprocessing(self, hand_key_point):
        none_indexes = [i for i, hand_position in enumerate(hand_key_point) if hand_position is None]
        for idx in none_indexes:
            hand_key_point[idx] = [0.0, 0.0, 0.0]
        base_keypoint = hand_key_point[0]
        base_keypoint_x, base_keypoint_y = base_keypoint[0], base_keypoint[1]
        keypoint_from_base = []
        for i, position in enumerate(hand_key_point):
            position_x, position_y = position[0], position[1]
            position_x -= base_keypoint_x
            position_y -= base_keypoint_y
            keypoint_from_base.append([position_x, position_y])

        return np.array(keypoint_from_base).ravel()


def draw_gesture(img,gesture,left_top):
    left, top = left_top
    action = None
    if gesture == "0":
        action = "nothing"
    elif gesture == "1":
        action = "opinion"
    elif gesture == "2":
        action = "question"
    cv2.putText(img, action, (300, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), lineType=cv2.LINE_AA)
    return img