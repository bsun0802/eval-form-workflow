import os
import sys
import io

from collections import namedtuple

import tensorflow as tf
import pandas as pd

from PIL import Image

sys.path.append('/home/ec2-user/obj-detection/models/research')
from object_detection.utils import dataset_util


flags = tf.app.flags
flags.DEFINE_string('img_path', '', 'Path to images')
flags.DEFINE_string('csv_input', '', 'Path to the CSV input')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS


LABELS=['checkbox1', 'comments1', 'barcode1']
def class_text_to_int(row_label):
    if LABELS.index(row_label) >= 0:
        return int(LABELS.index(row_label) + 1)

    
def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(x, gb.get_group(x)) for x in gb.groups.keys()]


def create_tf_example(group, path):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    image = Image.open(io.BytesIO(encoded_jpg))
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'

    xmins, xmaxs, ymins, ymaxs = [], [], [], []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        assert width == int(row['width'])
        assert height == int(row['height'])
        xmins.append(float(row['xmin']) / width)
        xmaxs.append(float(row['xmax']) / width)
        ymins.append(float(row['ymin']) / height)
        ymaxs.append(float(row['ymax']) / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

def main(_):
    path = FLAGS.img_path # path to train_imgs or eval_imgs

    examples = pd.read_csv(FLAGS.csv_input)
    grouped = split(examples, 'filename')  # group records by filename(image filename)
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    for group in grouped:
        tf_example = create_tf_example(group, path)
        writer.write(tf_example.SerializeToString())
    writer.close()

    output_path = FLAGS.output_path
    print('Successfully created the TFRecords: {}'.format(output_path))


if __name__ == '__main__':
    tf.app.run()


    
# To be used along with train_test_split.py
#train    
#python label_csv_to_tfRecord.py --csv_input=/home/ec2-user/obj-detection/annotations/coordinates_train.csv --img_path=/home/ec2-user/obj-detection/train_images --output_path=/home/ec2-user/obj-detection/annotations/coordinates_train.record

#eval
#python label_csv_to_tfRecord.py --csv_input=/home/ec2-user/obj-detection/annotations/coordinates_eval.csv --img_path=/home/ec2-user/obj-detection/eval_images --output_path=/home/ec2-user/obj-detection/annotations/coordinates_eval.record





# 2019/08/02
# python label_csv_to_tfRecord.py --csv_input=/home/ec2-user/v2-obj-detection/annotations/coordinates.csv --img_path=/home/ec2-user/v2-obj-detection/better-images --output_path=/home/ec2-user/v2-obj-detection/annotations/coordinates.record