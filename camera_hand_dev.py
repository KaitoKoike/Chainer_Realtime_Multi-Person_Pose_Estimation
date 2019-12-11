import cv2
import argparse
import json
import yaml
import chainer
import time
import requests
from pose_detector import PoseDetector, draw_person_pose
from hand_detector import HandDetector, draw_hand_keypoints
from gesture_recognizer import GestureRecognizer, draw_gesture, get_student_status
#import cupy as cp
#chainer.using_config('enable_backprop', False)
#pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
#cp.cuda.set_allocator(pool.malloc)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='hand detector')
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU ID (negative value indicates CPU)')
    parser.add_argument("--mode", "-m", type=str, default="camera", help="select whether you watch a result or not ")
    parser.add_argument("--cameraId", "-c", type=int, default=0,help="select an id of camera you will use")
    args = parser.parse_args()

    # load model
    hand_detector = HandDetector("handnet", "models/handnet.npz", device=args.gpu)
    pose_detector = PoseDetector("posenet", "models/coco_posenet.npz", device=args.gpu)
    right_gesture_recognizer = GestureRecognizer(model_path="models/right_gesture_recog_model.pkl")
    left_gesture_recognizer = GestureRecognizer(model_path="models/left_gesture_recog_model.pkl")
    cap = cv2.VideoCapture(args.cameraId)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("camera captured")

    while True:
        # get video frame
        ret, img = cap.read()
        if not ret:
            print("Failed to capture image")
            time.sleep(2)
            continue

        person_pose_array, _ = pose_detector(img)
        res_img = img[:]
        if args.mode == "camera":
            res_img = cv2.addWeighted(img, 0.6, draw_person_pose(img, person_pose_array), 0.4, 0)

        person_pose_array = sorted(person_pose_array, key=lambda x:x[0][0])
        for speaker_id,person_pose in enumerate(person_pose_array):
            unit_length = pose_detector.get_unit_length(person_pose)
            if args.mode == "camera":
                print(tuple(person_pose[0][:2]))
                cv2.putText(res_img, str(speaker_id), tuple(map(int,person_pose[0][:2])),cv2.FONT_HERSHEY_SIMPLEX,0.8, (0, 0, 0), lineType=cv2.LINE_AA)

            # hands estimation
            hands = pose_detector.crop_hands(img, person_pose, unit_length)
            hand_gesture_right = "0"
            hand_gesture_left = "0"
            if hands["left"] is not None:
                hand_img = hands["left"]["img"]
                bbox = hands["left"]["bbox"]
                hand_keypoints = hand_detector(hand_img, hand_type="left")
                res_img = draw_hand_keypoints(res_img, hand_keypoints, (bbox[0], bbox[1]))
                hand_gesture_left = left_gesture_recognizer(hand_keypoints, unit_length)
                if args.mode == "camera":
                    res_img = draw_gesture(res_img, hand_gesture_left, tuple(map(int,(person_pose[7][0], person_pose[7][1]))))

            if hands["right"] is not None:
                hand_img = hands["right"]["img"]
                bbox = hands["right"]["bbox"]
                hand_keypoints = hand_detector(hand_img, hand_type="right")
                res_img = draw_hand_keypoints(res_img, hand_keypoints, (bbox[0], bbox[1]))
                hand_gesture_right = right_gesture_recognizer(hand_keypoints,unit_length)
                if args.mode == "camera":
                    res_img = draw_gesture(res_img, hand_gesture_right, tuple(map(int,(person_pose[4][0], person_pose[4][1]))))
            print("speaker_id:",speaker_id," は，","右手: ",hand_gesture_right," 左手: ",hand_gesture_left,"です")
            student_status = get_student_status(hand_gesture_left,hand_gesture_right)
            data = json.dumps({"speaker_id":speaker_id,"student_status":student_status})
            message = yaml.dump({"data":data})
            query = {"message":message,'topic_name':'printeps/std_msgs/update_student_status'}
            requests.post("http://yamlab-Surface-Book-2.local:8080/publish",data=query)
        if args.mode == "camera":
            cv2.imshow("result", res_img)
        cv2.waitKey(10)

