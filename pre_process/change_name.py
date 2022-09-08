import os.path
import xml.dom.minidom


path = '/Users/zhangzixuan/Desktop/ssd300/VOCdevkit/VOC2007/Annotations/'

name = ['motorcycle', 'bicycle/motorcycle']

files = os.listdir(path) 
for xmlFile in files:
    print(xmlFile)
    if xmlFile == '.DS_Store':
        continue
    dom = xml.dom.minidom.parse(path + xmlFile)
    root = dom.documentElement
    newfilename = root.getElementsByTagName('name')
    for i, t in enumerate(newfilename):
        if t.firstChild.data == name[0]:
            newfilename[i].firstChild.data = name[1]
    with open(os.path.join(path, xmlFile), 'w') as fh:
        dom.writexml(fh)




