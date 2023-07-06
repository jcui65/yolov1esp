import argparse
import xml.etree.ElementTree as ET
import os

parser = argparse.ArgumentParser(description='Build Annotations.')
parser.add_argument('dir', default='..', help='Annotations.')#dir is VOCdevkit (until VOCdevkit)

sets = [('2007', 'train'), ('2007', 'val'), ('2007', 'test')]

classes_num = {'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3, 'bottle': 4, 'bus': 5,
               'car': 6, 'cat': 7, 'chair': 8, 'cow': 9, 'diningtable': 10, 'dog': 11,
               'horse': 12, 'motorbike': 13, 'person': 14, 'pottedplant': 15, 'sheep': 16,
               'sofa': 17, 'train': 18, 'tvmonitor': 19}#it is a dictionary


def convert_annotation(dir, year, image_id, f):
    in_file = os.path.join(dir, 'VOC%s/Annotations/%s.xml' % (year, image_id))#path concatenator, lock on the xml file
    tree = ET.parse(in_file)#get the
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        classes = list(classes_num.keys())#keys are the object classes, like dogs or cats, etc.
        if cls not in classes or int(difficult) == 1:#which means not passing the difficult ones?
            continue
        cls_id = classes.index(cls)#get the value of the dictionary!
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text),
             int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))#that 4 numbers!
        f.write(' ' + ','.join([str(a) for a in b]) + ',' + str(cls_id))#concatenate them, with comma in between
        #I think I will use that line of code later!#it seems it is just concatenating, right?

def _main(args):
    dir = os.path.expanduser(args.dir)#until VOCdevkit

    for year, image_set in sets:
        with open(os.path.join(dir, 'VOC%s/ImageSets/Main/%s.txt' % (year, image_set)), 'r') as f:
            image_ids = f.read().strip().split()#read first, then get rid of the spaces on both ends, then break into pieces
        with open(os.path.join(dir, '%s_%s.txt' % (year, image_set)), 'w') as f:#gonna write to 2007_train.txt
            for image_id in image_ids:#one after another, read one, write one, in train image set there are 2501 images
                f.write('%s/VOC%s/JPEGImages/%s.jpg' % (dir, year, image_id))#let convert_annotation function to add spaces
                convert_annotation(dir, year, image_id, f)#f is the file that I am gonna write annotations to
                f.write('\n')#don't forget to change the line to the next line!


if __name__ == '__main__':
    _main(parser.parse_args())
    # _main(parser.parse_args(['D:/Datasets/VOC/VOCdevkit']))
