import os

path="/Users/zhangzixuan/Desktop/newbdd/Annotations"
data = open(r"/Users/zhangzixuan/Desktop/newbdd/ImageSets/Main/train_val.txt","w", encoding='utf-8')

dirs = os.listdir(path)

for parent,dirnames,filenames in os.walk(path):
    for filename in sorted(filenames):
        filename=filename.partition(".")

        print (filename)
        data.writelines(filename[0]+ '\n')
data.close()




