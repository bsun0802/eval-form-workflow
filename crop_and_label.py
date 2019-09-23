import os
import sys
import shutil
import pickle
import sys
import time
import glob

import cv2
import numpy as np

from PIL import Image

label1 = 'checkbox1'
label2 = 'comments1'
label3 = 'barcode1'


def crop(name, img, label):
    showCrosshair = False
    fromCenter = False
    r = cv2.selectROI(name, img, fromCenter, showCrosshair)  # don't press 'c', return 0,0,0,0, re-drag the mouse should work
    y1, y2, x1, x2 = r[1], r[1]+r[3], r[0], r[0] + r[2]
    ROI = img[y1:y2, x1:x2]
    
    # visualize crop & label
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(img, label, (x1, y1), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

    height, width = img.shape[:2]
    record = f'{name},{x1},{y1},{x2},{y2},{label},{height},{width}\n'
    return ROI, record


def commit_crop(ROIs, records, im_name, labels):
    for i, label in enumerate(labels):
        cv2.imwrite(path_to_cropped + f'{label}_{im_name[:-4]}_crop_{i}.jpg', ROIs[i])
        coordinates.write(records[i])
    coordinates.flush()

    
# PATH 1
path = 'C:/Users/Bo.Sun/repos/objection-detection/04-labeling-job/'
impath = path + 'original-images/'
path_to_cropped = path + 'cropped/'
path_to_done = path + 'train/'
path_to_skip = path + 'skipped/'
path_to_marked = path + 'marked/'

# path_to_glob_tifs = 'E:/rx-eval-form/2019/*.tif'
path_to_glob_tifs = 'C:/Users/Bo.Sun/repos/objection-detection/04-labeling-job/original-images/*.tif'

for p in (path_to_cropped, path_to_done, path_to_skip, path_to_marked):
    if not os.path.exists(p):
        os.makedirs(p)

        
if not os.path.exists(impath):
    os.makedirs(impath)
M_START = time.time()
print('Start move images.')
if not glob.glob(path_to_glob_tifs):
    print('No tif found, assume all jpgs already')
else:
    for tif in glob.glob(path_to_glob_tifs):
        save_as = os.path.splitext(os.path.basename(tif))[0] + '.jpg'
        img = Image.open(tif)
        img.save(os.path.join(impath, save_as), quality=95)
        os.unlink(tif)
print(f'Done transform tif to jpg images.  {time.time() - M_START:.4f} s')

    
if os.listdir(impath):
    print(f'Found {len(os.listdir(impath))} Images')
    print('Start labeling..')
else:
    sys.exit(f'No Images found in {impath}, check your Input.')

    
# PATH 2
path_to_coordinates_file = path + 'coordinates.csv'
if os.path.exists(path_to_coordinates_file):
    coordinates = open(path_to_coordinates_file, 'a')
else:
    coordinates = open(path_to_coordinates_file, 'a')
    coordinates.write('filename,xmin,ymin,xmax,ymax,class,height,width\n')

# keep track of finished images
path_to_checkpoint = os.path.join(path, 'labeled_images.pkl')
if os.path.exists(os.path.join(path, 'labeled_images.pkl')):
    with open(path_to_checkpoint, 'rb') as f:
        labeled = pickle.load(f)
else:
    labeled = set()

    
for im_name in os.listdir(impath):
    if len(labeled) == len(os.listdir(impath)):
        sys.exit('all done!')
        
    if im_name.startswith('.') or im_name in labeled:
        continue
        
    ROIs, records, labels = [], [], []
    done = False  # done for the current image?
    img = cv2.imread(os.path.join(impath, im_name))
    clone = img.copy()
    cv2.namedWindow(im_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(im_name, 1200, 600)
    
    i = 0
    while not done:
        cv2.imshow(im_name, img)
        key_stroke = cv2.waitKey(0) & 0xFF
        
        if key_stroke == ord('1'):
            label = label1
            ROI, record = crop(im_name, img, label)  # click, drag and hit Enter to crop
            ROIs.append(ROI)
            records.append(record)
            labels.append(label)
            
        elif key_stroke == ord('2'):
            label = label2
            ROI, record = crop(im_name, img, label)
            ROIs.append(ROI)
            records.append(record)
            labels.append(label)
           
        elif key_stroke == ord('3'):
            label = label3
            ROI, record = crop(im_name, img, label)
            ROIs.append(ROI)
            records.append(record)
            labels.append(label)
           
        elif key_stroke == ord(' '):
            done = True
            if i == 0:
                shutil.copyfile(os.path.join(impath, im_name), os.path.join(path_to_skip, im_name))
            else:
                commit_crop(ROIs, records, im_name, labels)
                shutil.copyfile(os.path.join(impath, im_name), os.path.join(path_to_done, im_name))
            labeled.add(im_name)
            
        elif key_stroke == ord('r'):  # start-again
            ROIs, records, labels = [], [], []
            img = clone.copy()
            i = 0
        
        if key_stroke == ord('m'):  # mark an image
            shutil.copyfile(os.path.join(impath, im_name), os.path.join(path_to_marked, im_name))
        elif key_stroke == ord('x'):  # exit and save what's finished
            with open(path_to_checkpoint, 'wb') as f:
                pickle.dump(labeled, f)         
            coordinates.close()
            sys.exit(f'{len(labeled)} images finished.')

        i += 1
    cv2.destroyAllWindows()
