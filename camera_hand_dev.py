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
import cupy as cp
chainer.using_config('enable_backprop', False)
pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
cp.cuda.set_allocator(pool.malloc)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='hand detector')
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU ID (negative value indicates CPU)')
    parser.add_argument("--mode", "-m", type=str, default="camera", help="select whether you watch a result or not ")
    parser.add_argument("--hostname","-hn",type=str,default="surface01",help="hostname of discussion PC")
    args = parser.parse_args()

    # load model
    hand_detector = HandDetector("handnet", "models/handnet.npz", device=args.gpu)
    pose_detector = PoseDetector("posenet", "models/coco_posenet.npz", device=args.gpu)
    right_gesture_recognizer = GestureRecognizer(model_path="models/right_gesture_recog_model.pkl")
    left_gesture_recognizer = GestureRecognizer(model_path="models/left_gesture_recog_model.pkl")
    cap = cv2.VideoCapture("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)2592, height=(int)1458,format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv flip-method=0 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink")
    print("camera captured")
    i = 0
    while True:
        # get video frame
        ret, img = cap.read()
        i += 1
        if i % 2 != 0:
            print("pass")
            i = i % 2
            pass

        else:
            if not ret:
                print("Failed to capture image")
                time.sleep(2)
                continue
            print("start image read")
            height = img.shape[0]
            width = img.shape[1]
            half_size = (round(width/10),round(height/10))
            img = cv2.resize(img,half_size)
            person_pose_array, _ = pose_detector(img)
            res_img = img[:]
            if args.mode == "camera":
                res_img = cv2.addWeighted(img, 0.6, draw_person_pose(img, person_pose_array), 0.4, 0)

            discussant_pose_array = []
            discussant_status_dict = {}

            for speaker_id,person_pose in enumerate(person_pose_array):
                unit_length = pose_detector.get_unit_length(person_pose)
                print("speaker_id:", speaker_id, "は，unit_length:", unit_length)
                if 15 < unit_length < 1000:
                    discussant_pose_array.append(person_pose)

            discussant_pose_array = sorted(discussant_pose_array, key=lambda x: x[0][0])
            post_flag = False
            for speaker_id,person_pose in enumerate(discussant_pose_array):
                unit_length = pose_detector.get_unit_length(person_pose)
                if args.mode == "camera":
                    print("id書き込む座標:",tuple(person_pose[0][:2]))
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
                # print("speaker_id:", speaker_id, " は，", "右手: ", hand_gesture_right, " 左手: ", hand_gesture_left, "です")
                student_status = get_student_status(hand_gesture_left, hand_gesture_right)
                discussant_status_dict[str(speaker_id+1)] = "{0}".format(student_status)
                if student_status != "0":
                    post_flag = True
                
            message = """
            data: '{0}' 
            """.format(json.dumps(discussant_status_dict))
            query = {"message": message, 'topic_name': '/printeps/std_msgs/update_student_status'}
            hostname = args.hostname
            url = "http://"+hostname+".local:8080/publish"

            try:
                if post_flag :
                    requests.post(url, data=query)
            except Exception as e:
                print(e)
            if args.mode == "camera":
                cv2.imshow("result", res_img)
            print("終了")
        i = i % 2
        time.sleep(0.2)
        cv2.waitKey(10)



