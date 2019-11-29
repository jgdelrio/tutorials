"""
WIDER Face & Person Challenge dataset:
    - https://competitions.codalab.org/competitions/20146#participate
    - http://shuoyang1213.me/WIDERFACE/
"""
import os
import shutil
import cv2
import numpy as np
from PIL import Image
from xml.dom.minidom import Document


BASE_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), "../ComputerVision/data/wider_face"))


def transform_wider_set_to_darknet(input_path=BASE_FOLDER,
                                   output_folder="wider_face_darknet"):
    if not os.path.exists(input_path):
        raise IOError(f"The specified data folder '{input_path}' does not exists")
    # 1. Make directories
    base = os.path.join(input_path, output_folder)
    img_path = os.path.join(base, "jpeg_images")
    label_path = os.path.join(base, "labels")

    create_folders = [base, img_path, label_path]
    for folder in create_folders:
        if not os.path.exists(folder):
            os.mkdir(folder)

    # 2. Generate training images
    annotations_ref = os.path.join(input_path, "wider_face_train_bbx_gt.txt")
    with open(annotations_ref, 'r') as ann_file:
        lines = ann_file.readlines()
    n_lines = len(lines)

    # 2.a create image list
    img_list_ref = os.path.join(base, "img_list.txt")
    if not os.path.exists(img_list_ref):
        i = 0
        with open(img_list_ref, "w") as img_list:
            while i < n_lines:                              # each entry has...
                img_name = str(lines[i].rstrip('\n'))       # 1- the filename
                num_of_face = lines[i + 1]                  # 2- a number with the number of entries (lines)
                img_list.write(img_name)
                img_list.write('\n')
                i += int(num_of_face) + 2
    # 2.b copy images in WIDER_train dataset to dir 'JPEGImages'
    with open(img_list_ref, "r") as img_list:
        lines_in_img = img_list.readlines()
    img_original = os.path.join(input_path, "WIDER_train", "images")
    for line in lines_in_img:
        line = line.strip('\n')
        filepath = os.path.join(img_original, line)
        shutil.copy(filepath, img_path)

    # 3. Generate training labels
    # 3.a obtain the width and height of all train images
    img_size_ref = os.path.join(base, "img_size.txt")
    if not os.path.exists(img_size_ref):
        i = 0
        with open(img_size_ref, "a") as img_f:
            while i < len(lines_in_img):
                img_name = str(lines_in_img[i].rstrip('\n'))
                img_path = os.path.join(img_original, img_name)
                print(f"Image Path: {img_path}")
                nImg = np.array(Image.open(img_path))
                print(f"Image Shape: {nImg.shape}")
                img_f.write(str(nImg.shape[1]) + ' ')  # width
                img_f.write(str(nImg.shape[0]) + '\n')  # height
                i = i + 1

    # 4. Modify valid labels
    # 3.b generate the VOC-like labels
    i, j = 0, 0
    with open(img_size_ref, "r") as img_size:
        img_size_lines = img_size.readlines()
    while i < n_lines:
        num_face = lines[i + 1]
        img_name = lines[i].split('/')
        img_name = img_name[1][:-5]
        s_s = img_size_lines[j].split()
        imgW, imgH = list(map(float, s_s))

        # Ref to the image txt file within the labels folder in which we are writing the label info
        img_txt_ref = os.path.join(label_path, str(img_name) + ".txt")
        with open(img_txt_ref, "a") as f_label:
            for numface in range(int(num_face)):
                s = lines[i + 2 + numface].split()
                x, y, w, h = list(map(float, s[:4]))
                x = 0.001 if x <= 0.0 else x
                y = 0.001 if y <= 0.0 else y
                w = 0.001 if w <= 0.0 else w
                h = 0.001 if h <= 0.0 else h
                cx = x + 0.5 * w
                cy = y + 0.5 * h
                vocX = cx / imgW
                vocY = cy / imgH
                vocW = w / imgW
                vocH = h / imgH

                if vocX <= 0.0:
                    print('x' + str(img_name))
                if vocY <= 0.0:
                    print('y' + str(img_name))
                if vocW <= 0.0:
                    print('w' + str(img_name))
                if vocH <= 0.0:
                    print('h' + str(img_name))

                f_label.write('0' + ' ')
                f_label.write(str(vocX) + ' ')
                f_label.write(str(vocY) + ' ')
                f_label.write(str(vocW) + ' ')
                f_label.write(str(vocH) + ' ')
                f_label.write('\n')
        i += int(num_face) + 2
        j += 1

    # 5. Generate training images list
    train_ref = os.path.join(base, "train.txt")
    if not os.path.exists(train_ref):
        with open(train_ref, "a") as train_file:
            for line in lines_in_img:
                st = line.split('/')
                st = st[1]
                new_st = []
                new_st.append(st)
                new_st.insert(0, img_path)
                train_file.write(''.join(new_st))


def method_name(bboxes, filename, saveimg, voc_annotation_dir, lms, img_set):
    xmlpath = os.path.join(voc_annotation_dir, filename[:-3] + "xml")
    doc = Document()
    annotation = doc.createElement("annotation")
    doc.appendChild(annotation)
    folder = doc.createElement("folder")
    folder_name = doc.createTextNode("widerface")
    folder.appendChild(folder_name)
    annotation.appendChild(folder)
    filenamenode = doc.createElement("filename")
    filename_name = doc.createTextNode(filename)
    filenamenode.appendChild(filename_name)
    annotation.appendChild(filenamenode)
    source = doc.createElement("source")
    annotation.appendChild(source)
    database = doc.createElement("database")
    database.appendChild(doc.createTextNode("wider face Database"))
    source.appendChild(database)
    annotation_s = doc.createElement("annotation")
    annotation_s.appendChild(doc.createTextNode("PASCAL VOC2007"))
    source.appendChild(annotation_s)
    image = doc.createElement("image")
    image.appendChild(doc.createTextNode("flickr"))
    source.appendChild(image)
    flickrid = doc.createElement("flickrid")
    flickrid.appendChild(doc.createTextNode("-1"))
    source.appendChild(flickrid)
    owner = doc.createElement("owner")
    annotation.appendChild(owner)
    flickrid_o = doc.createElement("flickrid")
    flickrid_o.appendChild(doc.createTextNode("yanyu"))
    owner.appendChild(flickrid_o)
    name_o = doc.createElement("name")
    name_o.appendChild(doc.createTextNode("yanyu"))
    owner.appendChild(name_o)
    size = doc.createElement("size")
    annotation.appendChild(size)
    width = doc.createElement("width")
    width.appendChild(doc.createTextNode(str(saveimg.shape[1])))
    height = doc.createElement("height")
    height.appendChild(doc.createTextNode(str(saveimg.shape[0])))
    depth = doc.createElement("depth")
    depth.appendChild(doc.createTextNode(str(saveimg.shape[2])))
    size.appendChild(width)
    size.appendChild(height)
    size.appendChild(depth)
    segmented = doc.createElement("segmented")
    segmented.appendChild(doc.createTextNode("0"))
    annotation.appendChild(segmented)
    for i in range(len(bboxes)):
        bbox = bboxes[i]
        objects = doc.createElement("object")
        annotation.appendChild(objects)
        object_name = doc.createElement("name")
        object_name.appendChild(doc.createTextNode("face"))
        objects.appendChild(object_name)
        pose = doc.createElement("pose")
        pose.appendChild(doc.createTextNode("Unspecified"))
        objects.appendChild(pose)
        truncated = doc.createElement("truncated")
        truncated.appendChild(doc.createTextNode('1'))
        objects.appendChild(truncated)
        difficult = doc.createElement('difficult')
        difficult.appendChild(doc.createTextNode('0'))
        objects.appendChild(difficult)
        bndbox = doc.createElement('bndbox')
        objects.appendChild(bndbox)
        xmin = doc.createElement('xmin')
        xmin.appendChild(doc.createTextNode(str(bbox[0])))
        bndbox.appendChild(xmin)
        ymin = doc.createElement('ymin')
        ymin.appendChild(doc.createTextNode(str(bbox[1])))
        bndbox.appendChild(ymin)
        xmax = doc.createElement('xmax')
        xmax.appendChild(doc.createTextNode(str(bbox[0] + bbox[2])))
        bndbox.appendChild(xmax)
        ymax = doc.createElement('ymax')
        ymax.appendChild(doc.createTextNode(str(bbox[1] + bbox[3])))
        bndbox.appendChild(ymax)

        if img_set == "train":
            has_lm = doc.createElement('has_lm')

            if lms[i] == -1:
                has_lm.appendChild(doc.createTextNode('0'))
            else:
                has_lm.appendChild(doc.createTextNode('1'))
                lm = doc.createElement('lm')
                objects.appendChild(lm)

                x1 = doc.createElement('x1')
                x1.appendChild(doc.createTextNode(str(lms[i][0][0])))
                lm.appendChild(x1)

                y1 = doc.createElement('y1')
                y1.appendChild(doc.createTextNode(str(lms[i][0][1])))
                lm.appendChild(y1)

                x2 = doc.createElement('x2')
                x2.appendChild(doc.createTextNode(str(lms[i][1][0])))
                lm.appendChild(x2)

                y2 = doc.createElement('y2')
                y2.appendChild(doc.createTextNode(str(lms[i][1][1])))
                lm.appendChild(y2)

                x3 = doc.createElement('x3')
                x3.appendChild(doc.createTextNode(str(lms[i][2][0])))
                lm.appendChild(x3)

                y3 = doc.createElement('y3')
                y3.appendChild(doc.createTextNode(str(lms[i][2][1])))
                lm.appendChild(y3)

                x4 = doc.createElement('x4')
                x4.appendChild(doc.createTextNode(str(lms[i][3][0])))
                lm.appendChild(x4)

                y4 = doc.createElement('y4')
                y4.appendChild(doc.createTextNode(str(lms[i][3][1])))
                lm.appendChild(y4)

                x5 = doc.createElement('x5')
                x5.appendChild(doc.createTextNode(str(lms[i][4][0])))
                lm.appendChild(x5)

                y5 = doc.createElement('y5')
                y5.appendChild(doc.createTextNode(str(lms[i][4][1])))
                lm.appendChild(y5)

                visible = doc.createElement('visible')
                visible.appendChild(doc.createTextNode(str(lms[i][5])))
                lm.appendChild(visible)

                blur = doc.createElement('blur')
                blur.appendChild(doc.createTextNode(str(lms[i][6])))
                lm.appendChild(blur)
            objects.appendChild(has_lm)
    with open(xmlpath, "w") as annotation_file:
        annotation_file.write(doc.toprettyxml(indent=''))



def generate_voc_sets(img_set=["train", "val"], input_path=BASE_FOLDER):
    if isinstance(img_set, str):
        img_set = [img_set]

    if not os.path.exists(input_path + "/ImageSets"):
        os.mkdir(input_path + "/ImageSets")
    if not os.path.exists(input_path + "/ImageSets/Main"):
        os.mkdir(input_path + "/ImageSets/Main")

    gt_file_path = os.path.join(input_path, "wider_face_" + img_set + "_bbx_gt.txt")
    f = open(input_path + "/ImageSets/Main/" + img_set + ".txt", 'w')
    with open(gt_file_path, 'r') as gt_file:
        while (True):                               # and len(faces)<10
            filename = gt_file.readline()[:-1]
            if filename == "":
                break
            filename = filename.replace("/", "_")
            imgfilepath = filename[:-4]
            f.write(imgfilepath + '\n')
            numbbox = int(gt_file.readline())
            for i in range(numbbox):
                line = gt_file.readline()
    f.close()


def transform_wider_set_to_voc(img_sets=None,
                               input_path=BASE_FOLDER,
                               output_folder="wider_face_voc",
                               min_size=10, resized_dim=(48, 48)):
    """
    Transform to VOC
    :param img_sets:
    :param input_path:    input folder
    :param output_folder: output folder
    :param min_size:      minimum face size
    :return:
    """
    if not os.path.exists(input_path):
        raise IOError(f"The specified data folder '{input_path}' does not exists")
    # 1. Make directories
    output_base = os.path.join(input_path, output_folder)
    label_path = os.path.join(output_base, "labels")

    create_folders = [output_base, label_path]
    for folder in create_folders:
        if not os.path.exists(folder):
            os.mkdir(folder)

    if img_sets is None:
        img_sets = ["train", "test", "val"]
    for img_set in img_sets:
        convert_to_yolo = False
        convert_to_voc = True
        # Ref to where the images are currently stored:
        img_dir = os.path.join(input_path, "WIDER_" + img_set, "images")
        # Ref to the label file
        gt_label_file_path = os.path.join(input_path, img_set + "_label.txt")
        # Where we are going to store the processed images, annotations and labels
        images_dir = os.path.join(output_base, "jpeg_Images")
        voc_annotation_dir = os.path.join(output_base, "Annotations")
        labels_dir = os.path.join(output_base, "labels")

        img_sets_ref = os.path.join(output_base, "ImageSets")
        img_main_ref = os.path.join(output_base, "ImageSets", "Main")

        create_folders = [images_dir, img_sets_ref, img_main_ref]
        for folder in create_folders:
            if not os.path.exists(folder):
                os.mkdir(folder)
        if convert_to_yolo:
            if not os.path.exists(labels_dir):
                os.mkdir(labels_dir)
        if convert_to_voc:
            if not os.path.exists(voc_annotation_dir):
                os.mkdir(voc_annotation_dir)

        index = 0
        current_filename = ""
        bboxes = []
        lms = []
        with open(os.path.join(img_main_ref, img_set + ".txt"), "w") as f_set:
            with open(gt_label_file_path, "r") as gt_file:
                for line in gt_file:
                    # Read each line of the labels file
                    line = line.strip()

                    if line == "":
                        if len(bboxes) != 0:
                            method_name(bboxes, filename, save_img, voc_annotation_dir, lms, img_set)
                            cv2.imwrite(images_dir + "/" + filename, save_img)
                            imgfilepath = filename[:-4]
                            f_set.write(imgfilepath + '\n')
                            print("end!")
                        break

                    if line.startswith("#"):
                        # This line is a filename ref (folder/name.jpg  e.x: Parade/0_Parade_abc.jpg)
                        if index != 0 and convert_to_voc:
                            if len(bboxes) > 0:
                                method_name(bboxes, filename, save_img, voc_annotation_dir, lms, img_set)
                                cv2.imwrite(images_dir + "/" + filename, save_img)
                                imgfilepath = filename[:-4]
                                f_set.write(imgfilepath + '\n')
                            else:
                                print("no face")

                        current_filename = filename = line[1:].strip()
                        print(f"\r{str(index)}:{filename}\t\t\t")
                        index = index + 1
                        bboxes = []
                        lms = []
                        continue
                    else:
                        # Line with coordinates of boxes
                        imgpath = os.path.join(img_dir, current_filename)   # Ref to the img
                        img = cv2.imread(imgpath)                           # Read the image if available
                        if not img.data:                                    # Empty image
                            break
                        save_img = img.copy()
                        showimg = save_img.copy()
                        line = [float(x) for x in line.strip().split()]
                        if int(line[3]) <= 0 or int(line[2]) <= 0:
                            continue        # width or heght are zero
                        x, y, width, height = list(map(int, line[:4]))
                        bbox = (x, y, width, height)
                        x2 = x + width
                        y2 = y + height
                        if width >= min_size and height >= min_size:
                            # Only consider faces which dimensions are bigger than the minimum
                            bboxes.append(bbox)
                            if img_set == "train":
                                if line[4] == -1:
                                    lms.append(-1)
                                else:
                                    lm = []
                                    for i in range(5):
                                        x = line[4 + 3 * i]
                                        y = line[4 + 3 * i + 1]
                                        lm.append((x, y))
                                    lm.append(int(line[4 + 3 * i + 2]))
                                    lm.append(line[19])
                                    lms.append(lm)
                            cv2.rectangle(showimg, (int(x), int(y)), (int(x2), int(y2)), (0, 255, 0))
                        else:
                            # Images bellow the min
                            save_img[y:y2, x:x2, :] = (104, 117, 123)
                            cv2.rectangle(showimg, (x, y), (x2, y2), (0, 0, 255))

                        filename = filename.replace("/", "_")

                        if convert_to_yolo:
                            # Save the box center and its dimensions, relative to the img size
                            height, width = save_img.shape[:2]
                            txt_path = os.path.join(labels_dir, filename[:-3] + "txt")
                            with open(txt_path, 'w') as txt_file:
                                for i in range(len(bboxes)):
                                    bbox = bboxes[i]
                                    xcenter = (bbox[0] + bbox[2] * 0.5) / width
                                    ycenter = (bbox[1] + bbox[3] * 0.5) / height
                                    wr = bbox[2] * 1.0 / width
                                    hr = bbox[3] * 1.0 / height
                                    txt_line = f"0 {xcenter} {ycenter} {wr} {hr}\n"
                                    txt_file.write(txt_line)


if __name__ == "__main__":
    transform_wider_set_to_voc()
