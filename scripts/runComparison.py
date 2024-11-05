#PYTORCH_IMG_DIR="../pytorch-yolo-v3-master/imgs"
#VAL_IMG_DIR="../data/VOC2012/val"

#from resizer import down_sample
import subprocess
import os, sys
#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#1) Downsample list of val (HR) Image with filter
#subprocess.call("mkdir tmp1", shell=True)
#for filename in os.listdir(VAL_IMG_DIR):
#    if filename.endswith(".jpg") or filename.endswith(".png"): 
#        down_sample(VAL_IMG_DIR + '/' +  filename)
#        subprocess.call("mv *.jpg tmp1/", shell=True)

#2) Run through SRGAN (SR)
#subprocess.call("mv ../downsamplings/* ../data/test/SRF_4/data/", shell=True)
#subprocess.call("python ../test_benchmark.py", shell=True)
for root, dirs, files in os.walk('../downsampled/*.jpg'):
    for name in files:
        subprocess.call('../test_image.py --image_name' + os.path.join(root, name), shell=True)


#3) Run (SR) through yolo
#Put data into yolo data folder
#subprocess.call("mv ../downsamplings/ ../pytorch-yolo-v3-master/imgs/", shell=True)
#subprocess.call("python ../pytorch-yolo-v3-master/detect.py --images imgs --det det", shell=True)

#subprocess.call("rm tmp1 -rf", shell=True)

#Run yolo

#4) Run (HR) through yolo

#5) Compare labels



