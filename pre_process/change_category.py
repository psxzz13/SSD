
import os
import shutil

ann_filepath = r''
img_filepath = r''
img_savepath = r''
ann_savepath = r''
if not os.path.exists(img_savepath):
    os.mkdir(img_savepath)

if not os.path.exists(ann_savepath):
    os.mkdir(ann_savepath)
names = locals()
classes =  ['traffic sign', 'traffic light', 'truck', 'pedestrian', 'bus',
           'bicycle', 'other person', 'motorcycle','other vehicle', 'train', 'car']

for file in os.listdir(ann_filepath):
    print(file)


    if os.path.isdir(ann_filepath + file) or file == '.DS_Store':
        continue
    fp = open(ann_filepath + file, 'r')
    ann_savefile = ann_savepath + file
    fp_w = open(ann_savefile, 'w')
    lines = fp.readlines()
    print(lines)
    ind_start = []
    ind_end = []
    lines_id_start = lines[:]
    lines_id_end = lines[:]

    classes1 = '\t\t<name>car</name>\n'
    classes2 = '\t\t<name>traffic sign</name>\n'
    classes3 = '\t\t<name>traffic light</name>\n'
    classes4 = '\t\t<name>truck</name>\n'
    classes5 = '\t\t<name>pedestrian</name>\n'
    classes6 = '\t\t<name>bus</name>\n'
    classes7 = '\t\t<name>bicycle</name>\n'
    classes8 = '\t\t<name>other person</name>\n'
    classes9 = '\t\t<name>motorcycle</name>\n'
    classes10 = '\t\t<name>other vehicle</name>\n'



    while "\t<object>\n" in lines_id_start:
        a = lines_id_start.index("\t<object>\n")
        ind_start.append(a)
        lines_id_start[a] = "delete"

    while "\t</object>\n" in lines_id_end:
        b = lines_id_end.index("\t</object>\n")
        ind_end.append(b)
        lines_id_end[b] = "delete"

    i = 0
    for k in range(0, len(ind_start)):
        names['block%d' % k] = []
        for j in range(0, len(classes)):
            if classes[j] in lines[ind_start[i] + 1]:
                a = ind_start[i]
                for o in range(ind_end[i] - ind_start[i] + 1):
                    names['block%d' % k].append(lines[a + o])
                break
        i += 1
        print(names['block%d' % k])

    if ind_start == []:
       ind_start = [0]
    string_start = lines[0:ind_start[0]]
    if((file[2:4]=='09') | (file[2:4]=='10') | (file[2:4]=='11')):
       string_end = lines[(len(lines) - 11):(len(lines))]
    else:
       string_end = [lines[len(lines) - 1]]

    a = 0
    for k in range(0, len(ind_start)):
        if classes1 in names['block%d' % k]:
            a += 1
            string_start += names['block%d' % k]
        if classes2 in names['block%d' % k]:
            a += 1
            string_start += names['block%d' % k]
        if classes3 in names['block%d' % k]:
            a += 1
            string_start += names['block%d' % k]
        if classes4 in names['block%d' % k]:
            a += 1
            string_start += names['block%d' % k]
        if classes5 in names['block%d' % k]:
            a += 1
        if classes6 in names['block%d' % k]:
            a += 1
        if classes7 in names['block%d' % k]:
            a += 1

        if classes8 in names['block%d' % k]:
            a += 1
        if classes9 in names['block%d' % k]:
            a += 1

            string_start += names['block%d' % k]
    string_start += string_end
    for c in range(0, len(string_start)):
        fp_w.write(string_start[c])
    fp_w.close()
    if a == 0:
        os.remove(ann_savepath + file)
    else:
        name_img = img_filepath + os.path.splitext(file)[0] + ".jpg"
        shutil.copy(name_img, img_savepath)
    fp.close()


