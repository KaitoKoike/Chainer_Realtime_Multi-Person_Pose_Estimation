import sys
sys.path.append("../")
from pose_detector import PoseDetector, draw_person_pose
from hand_detector import HandDetector, draw_hand_keypoints
import cv2
import argparse
import chainer
import cupy as cp
import json
import os
import glob


chainer.using_config('enable_backprop', False)
pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
cp.cuda.set_allocator(pool.malloc)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='hand detector')
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU ID (negative value indicates CPU)')
    parser.add_argument("--precise", "-p", type=str, default=False, help="pose detector mode (precise or not)")
    parser.add_argument("--img_dir", "-id", type=str, default=None, help="image directory to make a data")
    args = parser.parse_args()

    # load model
    hand_detector = HandDetector("handnet", "../models/handnet.npz", device=args.gpu)
    pose_detector = PoseDetector("posenet", "../models/coco_posenet.npz", device=args.gpu, precise=args.precise)

    # train data set用の場所
    train_file = open("train.csv", "a")
    train_file.write("hand_position,right_hand,left_hand")

    # フォルダがなければ作成する
    if not os.path.exists("result_handimg"):
        os.mkdir("result_handimg")
    if not os.path.exists("hand_position_dataset"):
        os.mkdir("hand_position_dataset")
    if not os.path.exists("hand_img"):
        os.mkdir("hand_img")

    for image_path in glob.glob(args.img_dir+"*"):
        print(image_path)
        # get video frame
        img = cv2.imread(image_path)
        if "hidari" in image_path:
            img = cv2.flip(img, 1)

        person_pose_array, _ = pose_detector(img)
        res_img = cv2.addWeighted(img, 0.6, draw_person_pose(img, person_pose_array), 0.4, 0)
        hands_result = {"result": []}
        for i, person_pose in enumerate(person_pose_array):
            print(i, "人目")
            unit_length = pose_detector.get_unit_length(person_pose)
            person_hand = {"right": None, "left": None, "unit_length": unit_length}
            # hands estimation
            hands = pose_detector.crop_hands(img, person_pose, unit_length)

            if hands["left"]:
                hand_img = hands["left"]["img"]
                hand_file_name = image_path.split("/")[-1].replace(".png", "_lefthand.png")
                cv2.imwrite("hand_img/"+hand_file_name, hand_img)

                bbox = hands["left"]["bbox"]
                hand_keypoints = hand_detector(hand_img, hand_type="left")
                if hand_keypoints:
                    person_hand["left"] = [list(map(float, keypoint)) if keypoint else
                                           None
                                           for keypoint in hand_keypoints]

                res_img = draw_hand_keypoints(res_img, hand_keypoints, (bbox[0], bbox[1]))

            if hands["right"]:
                hand_img = hands["right"]["img"]
                hand_file_name = image_path.split("/")[-1].replace(".png", "_righthand.png")
                cv2.imwrite("hand_img/" + hand_file_name, hand_img)
                bbox = hands["right"]["bbox"]
                hand_keypoints = hand_detector(hand_img, hand_type="right")
                if hand_keypoints:
                    person_hand["right"] = [list(map(float, keypoint)) if keypoint else
                                            None
                                            for keypoint in hand_keypoints]

                res_img = draw_hand_keypoints(res_img, hand_keypoints, (bbox[0], bbox[1]))

            hands_result["result"].append(person_hand)

        # res_imgの保存
        result_img_path = image_path.split("/")[-1].replace(".png", "result.png")
        cv2.imwrite("result_handimg/"+result_img_path, res_img)

        # hand positionの保存
        hands_position_file_path = "hand_position_dataset/"+image_path.split("/")[-1].replace(".png", ".json")
        hands_file = open(hands_position_file_path, "w")
        json.dump(hands_result, hands_file)

        cv2.imshow("result_image", res_img)

        # 意見の時1，質問の時2, それ以外0
        if "1finger" in image_path:
            ob = 1
        elif "5finger" in image_path:
            ob = 2
        else:
            ob = 0

        train_line = "\n"+hands_position_file_path+","+str(ob)
        train_file.write(train_line)

        cv2.waitKey(1)
