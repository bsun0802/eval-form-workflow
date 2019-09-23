import os
from PIL import Image
import numpy as np
import json
import boto3
# from google.cloud import vision
import pytesseract
import zxing
import cv2
import time
import re

def tif_to_jpg(path_to_read, path_to_save, image_name, resize_perc = 1):
    """Convert .tif to .jpg, resize, and save. 
    
    Args:
        path_to_read(str): path to tif
        path_to_save(str): path to save jpg
        image_name(str): file name
        resize_perc(float): resize percentage, default is 1 which means not to resize
        
    Returns:
        NULL, save image.
    """
    if image_name.split(".")[-1].lower() in ['tif', 'tiff']:
        # read image
        img = Image.open(os.path.join(path_to_read, image_name))
        # resize
        wsize = int(img.size[0]*resize_perc)
        hsize = int(img.size[1]*resize_perc)
        img = img.resize((wsize, hsize))
        # save
        outfile = os.path.join(path_to_save, image_name[:-4]) + ".jpg"
        img.save(outfile)

        
def read_image_bytes(path_to_image, image_name):
    image_path = os.path.join(path_to_image, image_name)
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
    return image_bytes


def filter_od_result(result, threshold):
    '''Filter the Tensorflow object detection model results using a threshold.
    
    Args:
        result(list): Tensorflow object detection model inference results
        threshold(float): threshold of confidence score
        
    Returns:
        dict: model inference results filtered by the threshold. 
    
    '''

    boxes = np.squeeze(result['predictions'][0]['detection_boxes'])
    scores = np.squeeze(result['predictions'][0]['detection_scores'])
    classes = np.squeeze(result['predictions'][0]['detection_classes']).astype(int)
    
    confident = scores >= threshold
    
    scores = scores[confident].tolist()    
    classes = classes[confident].tolist()
    bboxes = boxes[confident].tolist()
    
    # inference contains staff with scores > threshold only.
    inference = dict(
        classes=classes,
        scores=scores,
        bboxes=bboxes
        )
    
    return inference


def region_extraction(imarr, bbox, region_type):
    """Extraction different regions, shrink/expand bboxes based on thier region types.
    
    Args:
        imarr(ndarray): eval form image in ndarray format
        bbox(list): coordinates
        region_type(str): region type - either question, answer, barcode, or comments
        
    Returns:
        ndarray: image region
    
    """

    ymin, xmin, ymax, xmax = bbox

    if region_type =='question':         
        ######## tune #############                                                                                                
        r_ymin = ymin
        # r_xmin = 0.02
        r_xmin = 0.008
        r_ymax = ymax
        r_xmax = min(xmin, 0.21)

    elif region_type == 'answer':
        ######## tune ############# 
        r_ymin = min(ymin+0.01, 1)
        r_xmin = max(xmax, 0)
        r_ymax = max(ymax-0.01, 0)
        r_xmax = min(xmax+0.09, 1)

    elif region_type == 'barcode':
        r_ymin = ymin
        r_xmin = xmin
        r_ymax = ymax
        r_xmax = xmax

    elif region_type == 'comments':
        ######## tune ############# 
        r_ymin = min(ymin+0.045, 1)
        r_xmin = min(xmin+0.005, 1)
        r_ymax = max(ymax-0.005, 0)
        r_xmax = max(xmax-0.005, 0)        

    # absolute coord    
    im_height, im_width = imarr.shape[:2]

    r_left = r_xmin * im_width
    r_right = r_xmax * im_width
    r_bottom = r_ymin * im_height
    r_top = r_ymax * im_height      

    # extract region
    region = imarr[int(r_bottom):int(r_top), int(r_left):int(r_right)]

    return(region)



def read_template(path_to_templates, departmentID):
    '''Read evaluation form templates using departmentID.
    
    Args: 
        path_to_templates(str): path to read eval form templates
        departmentID(str): 4-char departmentID, which we decoded from its barcode
    
    Returns:
        list: the template. If there is no departmentID, merge all templates and return. 
    
    '''
    template = []
    
    # templates mapping
    departments = {'fXEd': 'fixed', 'iANt': 'implants', 'tFOr': 'thermoform',
                   'cOMb': 'combo', 'fCAs': 'fullcast', 'bIOt': 'biotemps',
                   'vPSt': 'valplast', 'dNTr': 'dentures', 'pRTl': 'partials', 
                   'cOPg': 'copings' }   
    
    # remove wrong departmentIDs
    departmentID = list(set(departments.keys()) & set(departmentID))
    
    if len(departmentID) == 0:
        # read all templates
        departmentID = list(departments.keys())
                             
    # read the template
    for i in departmentID:
        with open(os.path.join(path_to_templates,'{}-template.json'.format(departments[i])), "rb") as infile:
            template += json.load(infile)['questions']
            infile.close()
    

    return template


def match_layout_answer(question, template, coord):
    """ Use the xmin to find out which column that the qustion belongs into, then match the answer by its question.
    
    Args:
        question(str): question
        template(list): a template with questions and answers
        coord(list): a list of coordinates, [ymin,xmin,ymax,xmax]
        
    Returns:
        str: the matched answer by its location (column)
    
    """ 
    
    # column boundary
    # ###### tune
    column1 = [0.2, 0.26]
    column2 = [0.3, 0.35]
    column3 = [0.4, 0.45]
    column4 = [0.5, 0.55]        

    xmin = coord[1]

    # match by column
    if xmin>column1[0] and xmin<column1[1]:
        column = 1
    elif xmin>column2[0] and xmin<column2[1]:
        column = 2
    elif xmin>column3[0] and xmin<column3[1]:
        column = 3
    elif xmin>column4[0] and xmin<column4[1]:
        column = 4
    else:
        return 'NA'

    answers = [x['answers'] for x in template if x['text']==question]
    
    if not answers:
        return 'NA'
    
    answer = answers[0][column-1]
    return answer 


def region_translation_barcode(inference, filename, imarr):
    """ Extract barcode regions, then use ZXing Barcode Reader to decode barcodes.
    
    Args:
        inference(dict): filtered model inference
        filename(str): filename
        imarr(ndarray): eval form image
        
    Returns:
        (dict, list): the dict contains barcodes text, coordinates, and confidence scores; the list contains departmentID

    """

    bb_barcode = [x for i,x in enumerate(inference['bboxes']) if inference['classes'][i]==3] 
    scores = [x for i,x in enumerate(inference['scores']) if inference['classes'][i]==3]
    decodes = []
    
    for b_barcode in bb_barcode:

        # crop barcode region
        b_region = region_extraction(imarr, b_barcode, 'barcode')
        cv2.imwrite('/tmp/{}_barcode.jpg'.format(filename),b_region)
    
        # decode barcode
        reader = zxing.BarCodeReader()
        barcode = reader.decode('/tmp/{}_barcode.jpg'.format(filename))         

        if barcode is None:
            decodes.append('NA')
        else:
            decodes.append(barcode.raw)            

    # save barcode
    bcode = dict(text = decodes,
                coords = bb_barcode, # (ymin,xmin,ymax,xmax)
                scores = scores
                )
    
    # departmentID
    departmentID = [x for x in decodes if len(x)==4]
    return (bcode, departmentID)


def read_far_right(c):
    ''' Read the far right selection only, if there are more than one answer per question.
    
    Args: 
        c(list[dict]): a list of checkbox dict (question text, answer text, coordinates, and cofidence scores)
    
    Returns:
        list[dict]: a list of filtered checkbox dict   
    
    ''' 
    
    c = sorted(c, key = lambda i: i['coords'][1],reverse=True)

    c2 = []
    for i in c:       
        if i['question'] not in [x['question'] for x in c2]:
            c2.append(i)           
    
    return c2


def visualization(imarr, r, path_to_vis_output, show=False):
    """ On each eval form, visualize its bboxes, scores, and its translated text. 
    
    Args:
        imarr(ndarray): eval form image
        r(dict): the pipeline json results
        show(bool): True - plot the visulization in-line, default is False - save the visualization in a folder
        
    Returns:
        NULL, save or plot the visualizations, depending on the input para
    
    """    
    image = imarr.copy()
    im_height, im_width = image.shape[:2]
    
    # 1 barcode
    if r['barcode'] != []:  
        b_text = r['barcode']['text']
        b_coords = r['barcode']['coords']
        b_scores = r['barcode']['scores']

        for i, coord in enumerate(b_coords):
            # bbox
            ymin, xmin, ymax, xmax = coord
            left, right, bottom, top = int(xmin * im_width), int(xmax * im_width), int(ymin * im_height), int(ymax * im_height)
            image = cv2.rectangle(image,(left,bottom),(right,top),(76,153,0),2)

            # text
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = b_text[i] + "_" + '{:.3f}'.format(b_scores[i])
            bottomLeftCornerOfText = (left,bottom-5)
            image = cv2.putText(image, text, bottomLeftCornerOfText, font, 0.8, (76,153,0), 2, cv2.LINE_AA)


    
    # 2 comments
#     if r['comments'] != []:    
#         c_text = r['comments']['text']
#         c_coords = r['comments']['coords']
#         c_scores = r['comments']['scores']

#         if c_text != 'No comments':    
#             # bbox
#             ymin, xmin, ymax, xmax = c_coords
#             left, right, bottom, top = int(xmin * im_width), int(xmax * im_width), int(ymin * im_height), int(ymax * im_height)
#             image = cv2.rectangle(image,(left,bottom),(right,top),(76,153,0),2)

#             # text
#             font = cv2.FONT_HERSHEY_SIMPLEX 
#             text = c_text + "_" + '{:.3f}'.format(c_scores)
#             bottomLeftCornerOfText = (left,bottom-5)
#             image = cv2.putText(image, text, bottomLeftCornerOfText, font, 0.8, (76,153,0), 2, cv2.LINE_AA)
    
    
    
    
    # 3 checkbox
    if r['checkbox'] != []:
        for cbox in r['checkbox']:
            q = cbox['question']
            a = cbox['answer']
            coords = cbox['coords']
            scores = cbox['scores']

            # bbox
            ymin, xmin, ymax, xmax = coords
            left, right, bottom, top = int(xmin * im_width), int(xmax * im_width), int(ymin * im_height), int(ymax * im_height)
            image = cv2.rectangle(image,(left,bottom),(right,top),(76,153,0),2)

            # text
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = q + ": " + a + "_" + '{:.3f}'.format(scores)
            bottomLeftCornerOfText = (left,bottom-5)
            image = cv2.putText(image, text, bottomLeftCornerOfText, font, 0.8, (76,153,0), 2, cv2.LINE_AA)
 
    
    if show:
        plt.imshow(image)
        plt.show()
    else:
        cv2.imwrite(os.path.join(path_to_vis_output, 'visualization_{}'.format(r['filename'])), image)
        print('Visualization: ' + os.path.join(path_to_vis_output, 'visualization_{}'.format(r['filename'])))
        
        
def region_translation_checkbox(inference, filename, imarr, path_to_templates, departmentID):
    """ Use localized checkboxes to extract their questions and answers regions. Then translate regions into text.
    
    Args:
        inference(dict): filtered model inference
        filename(str): filename
        imarr(ndarray): eval form image
        path_to_templates(str): path where tempaltes are saved into
        departmentID(str): department ID
    
    Returns:
        dict: the translated question text, answer text, coordinates, and cofidence scores
    
    """
    
    template = read_template(path_to_templates, departmentID)
    cboxes = []
    
    bb_checkbox = [x for i,x in enumerate(inference['bboxes']) if inference['classes'][i]==1]
    scores = [x for i,x in enumerate(inference['scores']) if inference['classes'][i]==1]
       
    for i,b_checkbox in enumerate(bb_checkbox):
        
        question=''
        answer=''
        
        
        # question                                                                                                
        q_region = region_extraction(imarr, b_checkbox, 'question')        
        
#         plt.imshow(q_region)
#         plt.show()        
        
        q_ocr = pytesseract.image_to_string(q_region)
        q_ocr = ' '.join(q_ocr.split())
        
#         print("Tesseract: ", q_ocr)
        
        question = text_processing(q_ocr)
        
#         print("Text_processing: ", question)
        
        if question:
                question = match_template(question, '', template)
                
#                 print("Match template: ", question)
        else:
            question = 'NA'
            
        if question == 'Please Contact Me':
            answer = 'Yes'
        else:
            # answer region: crop, ocr, text cleanning  
            a_region = region_extraction(imarr, b_checkbox, 'answer') 


#             plt.imshow(a_region)
#             plt.show()

            a_ocr = pytesseract.image_to_string(a_region)  
            a_ocr = ' '.join(a_ocr.split())

#             print("Tesseract: ", a_ocr)

            if not a_ocr:
                answer = match_layout_answer(question, template, b_checkbox)

#                 print('Match layout: ', answer)

            else:
                answer = text_processing(a_ocr)

#                 print("Text_processing: ", answer)

                if answer:
                    answer = match_template(question, answer, template)

#                     print("Match template: ", answer)

                else:
                    answer = match_layout_answer(question, template, b_checkbox)

#                     print('Match layout: ', answer)
                
            
        # save question and answer
        cbox = dict(question = question,
                    answer = answer,
                    coords = b_checkbox, # (ymin,xmin,ymax,xmax)
                    scores = scores[i]
                    )
        cboxes.append(cbox)
        
    return cboxes


def text_processing(text_ocr):
    'Take the first two recognized word that has len >= 2'
    allowed_chars = set()
    for i in list(range(65, 91))+list(range(97, 123)) + [32]: # [a-zA-Z] and ' '. 
        allowed_chars.add(chr(i))
    
    # expand the allowed chars
    allowed_chars.add('\n')
    
    text_ocr_ascii_only = ''.join(ch for ch in text_ocr if ch in allowed_chars)
    candidates = [word for word in re.split('\s+', text_ocr_ascii_only) if len(word) >= 2]
    
    if len(candidates) == 1:
        result = candidates[0].capitalize()
    elif len(candidates) > 1:
        result = candidates[0].capitalize() + ' ' + candidates[1].capitalize()
    else:
        result = ''
    return result



def match_template(question, answer, template):
    ''' Match the question/answer to the template's questions/answers, use edit distance and return the most similar one.
    
    Args:
        question(str): question, if answer is null, then it is supposed to match question only
        answer(str): answer, if question is null, then it is supposed to match answer only
        template(list): template
        
    Returns:
        str: the matched question/answer by template
    '''
        
    # match question
    if not answer:
        template_q = [x['text'] for x in template]  
        
        q_score = [(x, levenshtein_ratio_and_distance(question, x)) for x in template_q]
        q_score.sort(key=lambda elem: elem[1], reverse=True)

        q_min = q_score[0][0]  
        return q_min


    # match answer
    else:
        template_a = [x['answers'] for x in template if x['text']==question]
        # if cannot find a template
        if not template_a:
            def match_all_answers(answer):
                all_answer=['Acceptable','Bulky','Dark','Excellent','Good','Heavy','High','Ideal','Large','Light','Long','Loose','Low','No','Open','Over','Overground','Poor','Premature Post. Contact','Shallow','Short','Small','Subgingival','Supragingival','Thin','Tight','Too Little','Too Long','Too Much','Too Thick','Too Thin','Unacceptable','Unacceptably Long','Under','Under Contoured','Yes']
                which = np.argmax([levenshtein_ratio_and_distance(answer, ans) for ans in all_answers])
                return all_answer[which]
            return match_all_answer(answer)
        
        template_a = template_a[0]
        a_score = [(x, levenshtein_ratio_and_distance(answer, x)) for x in template_a]
        a_score.sort(key=lambda elem: elem[1], reverse=True)

        
        a_min = a_score[0][0]  
        return a_min
    

def levenshtein_ratio_and_distance(s, t):
    """ Compute the levenshtein distance ratio of similarity between two strings.
    For all i and j, distance[i,j] will contain the Levenshtein
    distance between the first i characters of s and the first j characters of t.
        
    Args:
        s(str): string of text
        t(str): string of text2
        
    Returns:
        float: the similiarity bwteen two strings
            
    """
    # Initialize matrix of zeros
    rows = len(s)+1
    cols = len(t)+1
    distance = np.zeros((rows,cols),dtype = int)

 

    # Populate matrix of zeros with the indeces of each character of both strings
    for i in range(1, rows):
        for k in range(1,cols):
            distance[i][0] = i
            distance[0][k] = k

 

    # Iterate over the matrix to compute the cost of deletions,insertions and/or substitutions    
    for col in range(1, cols):
        for row in range(1, rows):
            if s[row-1] == t[col-1]:
                cost = 0 # If the characters are the same in the two strings in a given position [i,j] then the cost is 0
            else:
                # the cost of a substitution is 2.             
                cost = 2
                
            distance[row][col] = min(distance[row-1][col] + 1,      # Cost of deletions
                                 distance[row][col-1] + 1,          # Cost of insertions
                                 distance[row-1][col-1] + cost)     # Cost of substitutions
    
    # Computation of the Levenshtein Distance Ratio
    Ratio = ((len(s)+len(t)) - distance[row][col]) / (len(s)+len(t))
    return Ratio


def handler():
    ''' Main function.
    
    '''    
    # define vars
    # path_to_tif = 'tif_improve'
    path_to_tif = 'tifs'
    path_to_image = 'jpgs'
    path_to_templates = 'templates'  
    endpoint_name = 'eval-form-tensorflow-serving-yuqi'
    
    runtime = boto3.client('runtime.sagemaker')

    threshold = 0.7
    
    import random
#     images = os.listdir(path_to_tif)
#     random.shuffle(images)
    images = ['TuyetHoa.Bui..1104612726.tif']
    for i, filename in enumerate(images[:150]):
        
        print('------------------')
        print('{}/{}'.format(i+1,len(images)))
        print(filename)
        
        if filename.split('.')[-1] in ['tif', 'tiff']:
            
            start = time.time() 
            
            # tif to jpg
            print('tif to jpg...')
            tif_to_jpg(path_to_tif, path_to_image, filename, resize_perc = 1)
            filename = filename[:-4]+'.jpg'
            
            # read image
            print('read image...')
            image_bytes = read_image_bytes(path_to_image, filename)
            imarr = cv2.imread(os.path.join(path_to_image, filename))  
            
            # OD model inference
            print('model inference...')
            response = runtime.invoke_endpoint(EndpointName=endpoint_name, Body=image_bytes, ContentType='application/x-image')
            inference0 = json.loads(response['Body'].read())            
            inference = filter_od_result(inference0, threshold)
            
            done1 = time.time()          
            
            # region OCR translation
            
            print('barcode OCR...')
            bcode, tempID = region_translation_barcode(inference, filename, imarr)
#             print('comments OCR...')
#             c_google = region_translation_comments(inference, filename, imarr)
            print('checkbox OCR...')
            cboxes = region_translation_checkbox(inference, filename, imarr, path_to_templates, tempID)
            
            # additional business rule
            cboxes = read_far_right(cboxes)
            
#             ocr_result = dict(filename = filename, barcode = bcode, checkbox = cboxes, comments = c_google)
            ocr_result = dict(filename = filename, barcode = bcode, checkbox = cboxes)
            done = time.time()
            

            print("Total time: {:.3f}s".format(done-start))
            print("\t Model time: {:.3f}s".format(done1-start))
            print("\t OCR time: {:.3f}s".format(done-done1))
        
            # print(ocr_result)
            
            
            # visulize ocr_result
            visualization(imarr, ocr_result, 'evaluation/2019-09-06', False) 
            
            # write results into json
            print('write json...')
            fname, a = '/tmp/pipeline_results.json', []
            if not os.path.isfile(fname):
                a.append(ocr_result)
                with open(fname, mode='w') as f:
                    f.write(json.dumps(a, indent=2))
            else:
                with open(fname) as feedsjson:
                    feeds = json.load(feedsjson)

                feeds.append(ocr_result)
                with open(fname, mode='w') as f:
                    f.write(json.dumps(feeds, indent=2))
            f.close()
            
            # write speed.csv
            print('write csv...')
            with open('/tmp/speed.csv', mode='a+') as file:
                file.write("{},{:.3f}s,{:.3f}s,{:.3f}s\n".format(filename, done-start, done1-start, done-done1))
            file.close()