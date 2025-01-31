cd models

wget http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/coco/pose_iter_440000.caffemodel
wget http://posefs1.perception.cs.cmu.edu/OpenPose/models/face/pose_iter_116000.caffemodel
wget http://posefs1.perception.cs.cmu.edu/OpenPose/models/hand/pose_iter_102000.caffemodel

python convert_model.py posenet pose_iter_440000.caffemodel coco_posenet.npz
python convert_model.py facenet pose_iter_116000.caffemodel facenet.npz
python convert_model.py handnet pose_iter_102000.caffemodel handnet.npz

cd ..