import codecs
import json
import xml
from glob import glob
import cv2
import shutil
import os
from sklearn.model_selection import train_test_split


labelme_path = "/Users/zhangzixuan/Desktop/train_json/"  
saved_path = "/Users/zhangzixuan/Desktop/bdd1"  
img_path = '/Users/zhangzixuan/Desktop/bdd100k/images/100k/train/'


if not os.path.exists(saved_path + "Annotations"):
    os.makedirs(saved_path + "Annotations")
if not os.path.exists(saved_path + "JPEGImages/"):
    os.makedirs(saved_path + "JPEGImages/")
if not os.path.exists(saved_path + "ImageSets/Main/"):
    os.makedirs(saved_path + "ImageSets/Main/")


files = glob(labelme_path + "*.json")
files = [i.split("/")[-1].split(".json")[0] for i in files]

for json_file_ in files:
    json_filename = labelme_path + json_file_ + ".json"
    #json_file = xml.load(open(json_filename, "r"))
    with open (json_filename) as fid:
        xml_str = fid.read()


    if ["labels"] not in xml_str:
        #os.remove(json_file)
        print(json_filename)


