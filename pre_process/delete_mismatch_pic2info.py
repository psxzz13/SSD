from PIL import Image
import os
import os.path
import numpy as np
import glob
import pandas as pd
import xml.etree.ElementTree as ET
import time

start = time.time()
n = 0
rootdir = r'/Users/zhangzixuan/Desktop/newbdd/JPEGImages/'
for parent, dirnames, filenames in os.walk(rootdir):
    for filename in filenames:
        currentPath = os.path.join(parent, filename) 
        picture_number = filename[:-4]
        print("the picture name is:" + filename)
        txt_path = r'/Users/zhangzixuan/Desktop/newbdd/Annotations/' 
        txt_file = os.listdir(txt_path)  
        if picture_number + '.xml' in txt_file:  
            pass
        else:
            print(currentPath)
            n += 1
            os.remove(currentPath)  
            print(picture_number)
            end = time.time()
end = time.time()
print("Execution Time:", end - start)

