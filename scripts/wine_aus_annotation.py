import json
import os
from os import listdir
from os.path import isfile, join
import xml.etree.ElementTree as ET
from absl import app, flags, logging
from absl.flags import FLAGS

flags.DEFINE_string('xml_dir', '/media/storage2/YOLO/WINE_AUS/annotations/xmls', 'path to label data')
flags.DEFINE_string('classes', '../data/classes/wine_aus.names', 'path to classes file')
flags.DEFINE_string('image_dir', "/media/storage2/YOLO/WINE_AUS/annotations/images", 'resize images to')
flags.DEFINE_string('anno_path_val', '../data/dataset/wine_aus.txt', 'path to annotation file')
flags.DEFINE_string('anno_path_json', '/media/storage2/YOLO/WINE_AUS/annotations/new_polys.json', 'path to label data')

def add_json_data(remove_existing=False):
    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    with open(FLAGS.anno_path_json, 'r') as fp:
        data = json.load(fp)
    if os.path.exists(FLAGS.anno_path_val) and remove_existing: os.remove(FLAGS.anno_path_val)

    with open(FLAGS.anno_path_val, 'a') as fp:
        for fname in data:
            image_path = os.path.join(FLAGS.image_dir, fname + ".jpg")
            print(image_path)
            annotation = image_path
            if not os.path.exists(image_path):
                continue
            else:
                for crop in data[fname]:
                    class_name = crop['crop_type']
                    if class_name == "rttrim":
                        class_name = 'crop'
                    if class_name not in class_names:
                        continue
                    class_ind = class_names.index(class_name)
                    bbox = crop['polygon']
                    xmin, ymin = bbox[0]
                    xmax, ymax = bbox[2]
                    annotation += ' ' + ','.join([str(xmin), str(ymin), str(xmax), str(ymax), str(class_ind)])
                fp.write(annotation + "\n")


def generate_dataset_file():
    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    xml_dir =  FLAGS.xml_dir
    input_images_dir = FLAGS.image_dir
    output = FLAGS.anno_path_val
    if os.path.exists(output): os.remove(output)

    input_image_paths = {}
    for root, dirs, files in os.walk(input_images_dir):
        for f in files:
            input_image_paths[os.path.splitext(f)[0]] = os.path.join(root, f)

    count = 0
    with open(output, 'a') as fp:
        for f in sorted(os.listdir(xml_dir)):
            print(f)
            d = f.split('.')[0]
            root = ET.parse(os.path.join(xml_dir, f))
            annotation = input_image_paths[d]
            for obj in root.findall('.//object'):
                # change name from e.g. 'crop', 'word_HUGH HAMILTON' to 'crop0', 'word0'
                class_name = obj.find('name').text.split('_')[0]
                if class_name == "rttrim":
                    class_name = 'crop'
                if class_name not in class_names:
                    continue
                class_ind = class_names.index(class_name)
                bb = obj.find('bndbox')
                xmin = int(bb.find('xmin').text)
                ymin = int(bb.find('ymin').text)
                xmax = int(bb.find('xmax').text)
                ymax = int(bb.find('ymax').text)
                annotation += ' ' + ','.join([str(xmin), str(ymin), str(xmax), str(ymax), str(class_ind)])
            fp.write(annotation + "\n")
            count += 1
    print(count)


def main(_argv):
    generate_dataset_file()
    add_json_data()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass