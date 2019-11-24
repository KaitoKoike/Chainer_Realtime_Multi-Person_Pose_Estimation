import cv2
import argparse
import chainer
from pose_detector import PoseDetector, draw_person_pose
from hand_detector import HandDetector, draw_hand_keypoints
from gesture_recognizer import GestureRecognizer, draw_gesture
#import cupy as cp
#chainer.using_config('enable_backprop', False)
#pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
#cp.cuda.set_allocator(pool.malloc)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='hand detector')
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU ID (negative value indicates CPU)')
    parser.add_argument("--precise", "-p", type=str, default=False, help="pose detector mode (precise or not)")
    parser.add_argument("--cameraId", "-c", type=int, default=0,help="select an id of camera you will use")
    args = parser.parse_args()

    # load model
    hand_detector = HandDetector("handnet", "models/handnet.npz", device=args.gpu)
    pose_detector = PoseDetector("posenet", "models/coco_posenet.npz", device=args.gpu, precise=args.precise)
    gesture_recognizer =GestureRecognizer(model_path="models/gesture_recog_model.pkl")
    cap = cv2.VideoCapture(args.cameraId)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("camera captured")

    while True:
        # get video frame
        ret, img = cap.read()

        if not ret:
            print("Failed to capture image")
            break

        person_pose_array, _ = pose_detector(img)
        res_img = cv2.addWeighted(img, 0.6, draw_person_pose(img, person_pose_array), 0.4, 0)
        for person_pose in person_pose_array:
            unit_length = pose_detector.get_unit_length(person_pose)

            # hands estimation
            hands = pose_detector.crop_hands(img, person_pose, unit_length)
            hand_gesture_right = 0
            hand_hesture_left = 0
            if hands["left"] is not None:
                hand_img = hands["left"]["img"]
                bbox = hands["left"]["bbox"]
                hand_keypoints = hand_detector(hand_img, hand_type="left")
                res_img = draw_hand_keypoints(res_img, hand_keypoints, (bbox[0], bbox[1]))
                hand_gesture_left = gesture_recognizer(hand_keypoints,unit_length)
                res_img = draw_gesture(res_img,hand_gesture_left,(bbox[0],bbox[1]))

            if hands["right"] is not None:
                hand_img = hands["right"]["img"]
                bbox = hands["right"]["bbox"]
                hand_keypoints = hand_detector(hand_img, hand_type="right")
                res_img = draw_hand_keypoints(res_img, hand_keypoints, (bbox[0], bbox[1]))
                hand_gesture_right = gesture_recognizer(hand_keypoints,unit_length)
                res_img = draw_gesture(res_img, hand_gesture_right, (bbox[0], bbox[1]))


        cv2.imshow("result", res_img)
        cv2.waitKey(1)

