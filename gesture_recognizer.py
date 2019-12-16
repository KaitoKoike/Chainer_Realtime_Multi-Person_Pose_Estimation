import pickle
import cv2
import numpy as np


class GestureRecognizer(object):
    def __init__(self, model_path=None):
        print('Loading GestureModel...')
        print(model_path)
        self.model = pickle.load(open(model_path, "rb"))

    def __call__(self, hand_keypoint, unit_length):
        if hand_keypoint is None:
            return "0"
        elif hand_keypoint[0] is not None:
            hand_keypoint = self.preprocessing_two(hand_keypoint, unit_length)

            gesture = self.model.predict([hand_keypoint])
            return gesture[0]
        else:
            print("key point is insufficient")
            return "0"

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

    def preprocessing_two(self,hand_key_point, unit_length):
        none_indexes = [i for i, hand_position in enumerate(hand_key_point) if hand_position is None]
        for idx in none_indexes:
            hand_key_point[idx] = [hand_key_point[0][0], hand_key_point[0][1], 0.0]
        base_keypoint = hand_key_point[0]
        each_vector = []
        for i, position_from in enumerate(hand_key_point):
            for j, position_to in enumerate(hand_key_point):
                if i != j:
                    each_vector.append([(position_to[0] - position_from[0]) / unit_length,
                                        (position_to[1] - position_from[1]) / unit_length])
        return np.array(each_vector).ravel()

def draw_gesture(img,gesture,left_top):
    left, top = left_top
    action = None
    if gesture == "0":
        action = "nothing"
    elif gesture == "1":
        action = "opinion"
    elif gesture == "2":
        action = "question"
    cv2.putText(img, action, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), lineType=cv2.LINE_AA)
    return img

def get_student_status(left_gesture, right_gesture):

    if (right_gesture == "1" and left_gesture == "0") or (right_gesture == "0" and left_gesture=="1"):
        return "1"
    elif (right_gesture == "2" and left_gesture == "0") or (right_gesture == "0" and left_gesture == "2"):
        return "2"
    else:
        return "0"
