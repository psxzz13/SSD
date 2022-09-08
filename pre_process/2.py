import os
path="/Users/zhangzixuan/Desktop/newbdd/Annotations/"
data = open(r"/Users/zhangzixuan/Desktop/newbdd/ImageSets/Main/train.txt","w", encoding='utf-8')

#file1 = open(r'/Users/zhangzixuan/Desktop/xml.txt', 'r')
file1 = open(r'/Users/zhangzixuan/Desktop/alltrain.txt','r')

file2 =  open(r'/Users/zhangzixuan/Desktop/newbdd/ImageSets/Main/train_val.txt', 'r')

sames = set(file1).intersection(file2)

for parent,dirnames,filesames in os.walk(path):
    for filename in sorted(sames):
        filename=filename.partition(".")
        print (filename)
        data.writelines(filename[0])
data.close()