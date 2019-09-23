import os
import glob
import argparse
import time
import shutil
from datetime import datetime
import pandas as pd
import cv2
import numpy as np
import tensorflow as tf

from utils import label_map_util
class TFObjectDetector:
    def __init__(self, path_to_model):
        self.PATH_TO_MODEL = path_to_model
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.PATH_TO_MODEL, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
            self.boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
            self.scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
            self.classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            self.num = self.detection_graph.get_tensor_by_name('num_detections:0')
        self.sess = tf.Session(graph=self.detection_graph)
    
    def detect(self, img):
        boxes, scores, classes, num = self.sess.run(
            [self.boxes, self.scores, self.classes, self.num],
            feed_dict={self.image_tensor: img}
        )
        return boxes, scores, classes, num


def object_detection(im_path, detector, threshold):
    img = cv2.imread(im_path)
    height, width = img.shape[:2]
    
    img_expanded = np.expand_dims(img, 0)
    boxes, scores, classes, num = detector.detect(img_expanded)
    boxes, scores, classes = np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes)

    confident = scores >= threshold
    bboxes = boxes[confident]
    cropped_coordinates = (bboxes * [height, width, height, width]).astype(int)
    cropped_regions = [img[box[0]:box[2], box[1]:box[3]] for box in cropped_coordinates]
    
    # inference contains staff with scores > threshold only.
    inference = dict(
        classes=[label_map[class_int] for class_int in classes[confident]],
        scores=scores[confident],
        bboxes=bboxes,
        cropped_coordinates=cropped_coordinates,
        cropped_regions=cropped_regions)
    return inference


def visualize_bboxes(im_path, inference):
    img = cv2.imread(im_path)
    filename = os.path.basename(im_path)
    
    print(f'{len(inference["cropped_regions"])} bboxes detected for {filename}')
    
    for idx, score in enumerate(inference['scores']):
        if args.no_save_crop:
            out_fn = '{a}_{b}_{c}'.format(
                a=_coords_to_str(inference['cropped_coordinates'][idx]),
                b=inference['classes'][idx],
                c=filename
            )
            cv2.imwrite(os.path.join(path_to_cropped_region, out_fn), inference['cropped_regions'][idx])
            print(f'Cropped regions saved to {path_to_cropped_region}')

        
        btm, left, top, right = inference['cropped_coordinates'][idx]
        bbox = cv2.rectangle(img,(left,btm),(right,top),(0,0,255),3)

        # add display_name and score to bbox
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = inference['classes'][idx] + ' ' + str(int(inference['scores'][idx]*100)) + '%'
        bottomLeftCornerOfText = (left,btm-5)
        bbox = cv2.putText(bbox, text, bottomLeftCornerOfText, font, 0.7,(0,0,255),2, cv2.LINE_AA)

    
    cv2.imwrite(os.path.join(path_to_inference_result, f'prediction_{filename}'), bbox)
    print(f'Image with bboxes saved to {path_to_inference_result}\n')

def _coords_to_str(coord):
    return ','.join(map(str, coord))
   
def parse_args():
    parser = argparse.ArgumentParser(description='Object Detection Inference')
    parser.add_argument('eval_images', help='path to eval_images')
    parser.add_argument('output_path', help='path to save predictions pkl')
    parser.add_argument('--threshold', default=.5, type=float, help='keep detection score > threshold')
    parser.add_argument('--no_save_crop', action='store_false')
    parser.add_argument('--model', help='path to inference graph')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    # initialize ObjectDetector, load the trained model. 
    if args.model:
        path_to_model = args.model
    else:
        path_to_model = '/home/ec2-user/obj-detection/output/model/frozen_inference_graph.pb'
    
    path_to_pbtxt = '/home/ec2-user/obj-detection/annotations/rx_form_3.pbtxt'
    for p in (path_to_model, path_to_pbtxt):
        if not os.path.exists(p):
            print(f'Wrong Input {p}')

    # path_to_eval_images = '/home/ec2-user/obj-detection/eval_images/'
    path_to_eval_images = args.eval_images
    path_to_cropped_region = os.path.join(args.output_path, 'd_bboxes')
    path_to_inference_result = os.path.join(args.output_path, 'd_results')  # visualization OD results on eval_images/
    path_to_prediction_log = os.path.join(args.output_path, 'predictions.pkl')
    for p in (path_to_cropped_region, path_to_inference_result):
        if not os.path.exists(p):
            os.makedirs(p)
            print(f'Create dir: {p}')
          
    category_index = label_map_util.create_category_index_from_labelmap(path_to_pbtxt, use_display_name=True)
    label_map = {i: obj['name'] for i, obj in category_index.items()}
    
    detector = TFObjectDetector(path_to_model=path_to_model)
    
    test_images = glob.glob(path_to_eval_images + '*')
    print(' [ '+datetime.now().strftime("%b %d %H:%M:%S")+' ]  ')
    print('Initializing inference graph, label map, test images..Done')
    print(f'Found {len(test_images)} Test images.')
    print(f'Start Object Detection with score >= {args.threshold}\n\n')
    if os.path.exists(path_to_prediction_log):
        os.unlink(path_to_prediction_log)
    records = []
    for im_path in test_images:
        filename = os.path.basename(im_path)
        print('['+datetime.now().strftime("%b %d %H:%M:%S")+']' + f'Detecting: {filename}')
        start_od = time.time()
        inference = object_detection(im_path, detector, args.threshold)
        speed = time.time() - start_od
        if len(inference['scores']) > 0:
            record = [filename, inference['classes'], inference['scores'], inference['cropped_coordinates'], speed]
            visualize_bboxes(im_path, inference)
        else:
            print(f'No bbox dected for {filename}, original image will be saved to result.')
            record = [filename] + [np.nan]*4
            shutil.copyfile(im_path, os.path.join(path_to_inference_result, f'prediction_{filename}'))
        records.append(record)
    
    df = pd.DataFrame(records, columns=['filename', 'class', 'score', 'coord(ymin, xmin, ymax, xmax)', 'speed'])
    df.to_pickle(path_to_prediction_log)

    
    
    
# python inference_v2.py /home/ec2-user/obj-detection/test-08-08/eval_images/ /home/ec2-user/obj-detection/test-08-08/ --model=/home/ec2-user/v2-obj-detection/model-01-export/frozen_inference_graph.pb