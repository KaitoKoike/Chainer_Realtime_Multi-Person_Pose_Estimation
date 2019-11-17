import sys
sys.path.append("../")
from pose_detector import PoseDetector, draw_person_pose
from hand_detector import HandDetector, draw_hand_keypoints
import cv2
import argparse
import chainer
import cupy as cp
import time
import json

chainer.using_config('enable_backprop', False)
pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
cp.cuda.set_allocator(pool.malloc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='hand detector')
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU ID (negative value indicates CPU)')
    parser.add_argument("--precise", "-p", type=str, default=False, help="pose detector mode (precise or not)")
    parser.add_argument("--video_path", "-vp", type=str, default=None, help="video path to make a data")
    args = parser.parse_args()

    # load model
    hand_detector = HandDetector("handnet", "../models/handnet.npz", device=args.gpu)
    pose_detector = PoseDetector("posenet", "../models/coco_posenet.npz", device=args.gpu, precise=args.precise)

    # ビデオをキャプチャーして解析し，保存する
    cap = cv2.VideoCapture(args.video_path)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        sys.exit()
    train_file = open("train.csv", "a")
    train_file.write("hand_position,right_hand,left_hand")

    while True:

        # get video frame
        ret, img = cap.read()

        if not ret:
            print("Failed to capture image")
            break

        person_pose_array, _ = pose_detector(img)
        res_img = cv2.addWeighted(img, 0.6, draw_person_pose(img, person_pose_array), 0.4, 0)
        hands_result = {"result": []}
        for i, person_pose in enumerate(person_pose_array):
            person_hand = {"right": None, "left": None}
            unit_length = pose_detector.get_unit_length(person_pose)

            # hands estimation
            hands = pose_detector.crop_hands(img, person_pose, unit_length)
            hands_result["result"].append(hands)
            if hands["left"] is not None:
                hand_img = hands["left"]["img"]
                bbox = hands["left"]["bbox"]
                hand_keypoints = hand_detector(hand_img, hand_type="left")
                person_hand["left"] = hand_keypoints
                res_img = draw_hand_keypoints(res_img, hand_keypoints, (bbox[0], bbox[1]))

            if hands["right"] is not None:
                hand_img = hands["right"]["img"]
                bbox = hands["right"]["bbox"]
                hand_keypoints = hand_detector(hand_img, hand_type="right")
                person_hand["right"] = hand_keypoints
                res_img = draw_hand_keypoints(res_img, hand_keypoints, (bbox[0], bbox[1]))

            hands_result["result"].append(person_hand)

        now_time = time.time()
        cv2.imshow("result", res_img)

        result_file_path_name = "result_images/result_hand"+str(now_time)+".png"
        hands_position_file_path = "result_hand_position/result_hand_pos"+str(now_time)+".json"
        hands_file = open(hands_position_file_path, "w")
        json.dump(hands_result, hands_file)

        cv2.imshow("result_image", res_img)
        print("これは意見なら1，質問なら2，それ以外なら0を記入")
        print("右")
        train_right = input()
        print("左")
        train_left = input()
        train_line = "\n"+hands_position_file_path+","+train_right+","+train_left
        train_file.write(train_line)

        ret = cv2.imwrite(result_file_path_name, res_img)

        if not ret:
            print("fail to save this image")
            break
        cv2.waitKey(1)
