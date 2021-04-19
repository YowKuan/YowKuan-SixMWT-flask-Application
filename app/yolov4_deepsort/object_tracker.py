import os
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
import subprocess
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import app.yolov4_deepsort.core.utils as utils
from app.yolov4_deepsort.core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from app.yolov4_deepsort.core.config import cfg
from PIL import Image
#cv2 version = 4.1.1
import cv2

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from imutils.video import FileVideoStream

# deep sort imports
from app.yolov4_deepsort.deep_sort import preprocessing, nn_matching
from app.yolov4_deepsort.deep_sort.detection import Detection
from app.yolov4_deepsort.deep_sort.tracker import Tracker
from app.yolov4_deepsort.tools import generate_detections as gdet
from app.yolov4_deepsort import speed_dis_calculation
#json import
import json
import pandas
import csv
import os.path

def main_normal(_argv):
    # Definition of the parameters
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 1.0
    
    # initialize deep sort
    model_filename = './app/yolov4_deepsort/model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # initialize tracker
    tracker = Tracker(metric)

    #
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print('success!!!')
    except:
    # Invalid device or cannot modify virtual devices once initialized.
        pass

    # initialize the total number of objects that have moved either up or down
    y_list = [0]
    pre_dis = 0
    pre_dis_4 = 0
    aft_dis = 0
    aft_dis_4 = 0
    speed_5s = 0
    speed_4s = 0
    distance = 0
    patient_id = 0
    start_from_bottom = 0

    #count_list
    speed = 0
    count_adjust =[]
    sign_list = []

    #create speed record list
    speed_record = []
    speed_record_time =[]
    time_record = []
    time_walkrounds = []
    every_3m_time = []
    #record distance at 15s(Initialize)
    D_1q = 0
    D_2q = 0
    # load configuration for object detector
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size

    with open('./app/yolov4_deepsort/vid_route.json', newline='') as routefile:
            viddd = json.load(routefile)
            vidd_path = viddd["Video route"]
    video_path = str(vidd_path)
    video_name = video_path.split('/')[-1].split('.')[0]
    Group = video_name.split('-')[0]
    ID = video_name.split('_')[0]
    Dt = video_name.split('_')[1]
    Date = (Dt[0:4]+'-'+Dt[4:6]+'-'+Dt[6:8])
    print('Group:', Group, 'ID:', ID, 'Date:', Date)
    #0322
    bound_list = [185, 235, 310, 435, 700, 700, 435, 310, 235, 185]*50
    s=0
    dis_counting = [0]
    distance = 0



    #try to increase eff with queue here

    # load tflite model if flag is set
    if FLAGS.framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
    # otherwise load standard tensorflow saved model
    else:
        saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']

    # begin video capture
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)
    current_frame = int(vid.get(cv2.CAP_PROP_POS_MSEC))

    out = None
    sample_rate = 5
    s_frame_count = 1
    total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    #total_frames = 1000
    vid_fps = 30
    
    
    total_time = total_frames/vid_fps
    
    
    # get video ready to save locally if flag is set
    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))


    #For video, while video is running
    for fno in range(0, total_frames, sample_rate):
        vid.set(cv2.CAP_PROP_POS_FRAMES, fno)
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
            s_frame_count+=1
        else:
            print('Video has ended or failed, try a different video format!')
            break
    
    
        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()
        vid_time = round(s_frame_count*sample_rate/vid_fps, 1)
        
        
        
        # run detections on tflite if flag is set
        if FLAGS.framework == 'tflite':
            interpreter.set_tensor(input_details[0]['index'], image_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            # run detections using yolov3 if flag is set
            if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
                boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
            else:
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
        else:
            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )

        # convert data to numpy arrays and slice out unused elements
        num_objects = valid_detections.numpy()[0]
        bboxes = boxes.numpy()[0]
        bboxes = bboxes[0:int(num_objects)]
        scores = scores.numpy()[0]
        scores = scores[0:int(num_objects)]
        classes = classes.numpy()[0]
        classes = classes[0:int(num_objects)]

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(bboxes, original_h, original_w)

        # store all predictions in one parameter for simplicity when calling functions
        pred_bbox = [bboxes, scores, classes, num_objects]

        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # by default allow all classes in .names file
        #allowed_classes = list(class_names.values())
        
        # custom allowed classes (uncomment line below to customize tracker for only people)
        allowed_classes = ['person']

        # loop through objects and use class index to get class name, allow only classes in allowed_classes list
        names = []
        deleted_indx = []
        for i in range(num_objects):
            class_indx = int(classes[i])
            class_name = class_names[class_indx]
            if class_name not in allowed_classes:
                deleted_indx.append(i)
            else:
                names.append(class_name)
        names = np.array(names)
        count = len(names)
        if FLAGS.count:
            cv2.putText(frame, "Objects being tracked: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
            print("Objects being tracked: {}".format(count))
        # delete detections that are not in allowed_classes
        bboxes = np.delete(bboxes, deleted_indx, axis=0)
        scores = np.delete(scores, deleted_indx, axis=0)

        # encode yolo detections and feed to tracker
        features = encoder(frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

        #initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima supression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]       

        # Call the tracker
        tracker.predict()
        tracker.update(detections)
        
        # update tracks
        for track in tracker.tracks:

            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr()
            x_middle = int(bbox[0])//2 + int(bbox[2])//2
            y_middle = int(bbox[1])//2 + int(bbox[3])//2
            

            #print(track.track_id,":", x_middle, y_middle)
            
            if vid_time <1 and 1050<x_middle<1150 and y_middle >100:
                patient_id = track.track_id
                print("Patient found! ID=", patient_id)
            if vid_time > 3 and speed_5s == 0 and 1050<x_middle<1150 and y_middle >100:
                patient_id = track.track_id
                print("Patient tracked! ID=", patient_id)
            

            if 800>y_middle>400:
                y_feet = int(y_middle*1.5)
            elif 399>y_middle>300:
                y_feet = int(y_middle*1.38)
            else:
                y_feet = int(y_middle*1.50)
            
            class_name = track.get_class()
            if 1050 < x_middle < 1300 and y_feet > 150 and (track.track_id==patient_id):
            #if 1050 < x_middle < 1250 and y_feet > 150:
                y_list.append(y_feet)  
                #print(y_feet)
            if (y_list[-1]-bound_list[s])*(dis_counting[-1]-bound_list[s])<0:
                dis_counting.append(y_list[-1])
                if dis_counting[-1]>800:
                    while bound_list[s]!= 700:
                        s+=1
                #print("current boundary:", bound_list[s], dis_counting[-1])
                while 400<dis_counting[-1]<800 and abs(bound_list[s]-dis_counting[-1])>200:
                    s+=1
                while dis_counting[-1]<400 and abs(bound_list[s]-dis_counting[-1])>100:
                    s+=1
                s+=1
                
                distance+=3
                if not every_3m_time or vid_time != every_3m_time[-1]:
                    every_3m_time.append(vid_time)

        # draw bbox on screen
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]
           # cv2.circle(frame,((int(bbox[0])+int(bbox[2])//2), (int(bbox[1])+int(bbox[3]))//2), 100, (0, 0, 255), 100)
            
            cv2.circle(frame,(x_middle, y_feet), 5, (0, 0, 255), 2)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
            cv2.putText(frame, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
            #1.5m line
            cv2.line(frame, (1050, 700), (1150, 700), (255, 0, 0), 2)
            #4.5m line
            cv2.line(frame, (1050, 435), (1150, 435), (255, 0, 0), 2)
            #7.5m line
            cv2.line(frame, (1050, 310), (1150, 310), (255, 0, 0), 2)
            #10.5m line
            cv2.line(frame, (1050, 235), (1150, 235), (255, 0, 0), 2)
            #13.5m line
            cv2.line(frame, (1050, 185), (1150, 185), (255, 0, 0), 2)
        
            
        
        if vid_time % 5 == 0:
            pre_dis = aft_dis
            aft_dis = distance
            speed_5s = (aft_dis - pre_dis)/5
            speed_record.append(round(speed_5s,2))
            speed_record_time.append(vid_time)

        if vid_time % 4 == 0:
            pre_dis_4 = aft_dis_4
            aft_dis_4 = distance
            speed_4s = (aft_dis_4 - pre_dis_4)/4
        


        # if enable info flag then print details about each track
        if FLAGS.info:
            print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))
        info = [
        ("Total Distance", round(distance), "m"),
        #("Current Frame", s_frame_count),
        ("Time", round(vid_time, 1), "s"),
        #("s_1q", round(D_1q/15, 2)),
        #("s_2q", round(D_2q/15, 2)),
        ("Speed_5s", round(speed_5s, 2), "m/s"),
        ("Speed_4s", round(speed_4s, 2), "m/s")
        #("Last5_Speed", round(last5_speed, 1))
        ]

        for (i, (k, v, q)) in enumerate(info):
            text = "{}: {} {}".format(k, v, q)
            cv2.putText(frame, text, (50, 800-i*40), cv2.FONT_HERSHEY_SIMPLEX,
  0.6, (0, 255, 220), 2)   
      
        # calculate frames per second of running detections
        fps = 1.0 / (time.time() - start_time)
        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        if not FLAGS.dont_show:
            cv2.imshow("Output Video", result)
        
        # if output flag is set, save video file
        if FLAGS.output:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    
    print("The total walking distance is:", distance, "m.")  
    video_length_count = s_frame_count*sample_rate/30
    average_speed = round(distance/video_length_count, 2)
    print("The average speed is:", average_speed, "m/s")
    print("Video_length:", round(video_length_count, 2), "s")
    print("Skipping Frame: ", sample_rate)
    print("your speed record is:", speed_record)
    equ_distance = round((6*distance/(vid_time/60)), 2)
    print("Equally, you walked",equ_distance, "m")
    print('Every_3m_result is:', every_3m_time)
    
    #generate walk result without processing
    walk_result = {}
    walk_result = {'Distance': distance, 'Average Speed': average_speed, '6min equ dis':equ_distance,
     'Video length':round(video_length_count, 2)}
    with open('./app/yolov4_deepsort/walk_result.json', 'w') as outfile:
        json.dump(walk_result, outfile)

    #generate the processed speed, round results
    #20210310
    speed_list, dis_dict = speed_dis_calculation.speed_dis_calc(speed_record, speed_record_time, Group, video_name, Date)
    round_record = speed_dis_calculation.round_calculation(speed_record, Group, video_name, Date)
    dis_speed = speed_dis_calculation.every_three_calculation(every_3m_time, Group, video_name, Date)
    #print("The converted round record is:", round_record)



def compare_walk_dis(p_height, p_weight, p_age):
    should_walk = 7.57*p_height -5.02*p_age -1.76*p_weight -309
    with open('./app/yolov4_deepsort/walk_result.json', newline='') as jsonfile:
        data = json.load(jsonfile)
        walk_distance = data["Distance"]
        equiv_distance = data["6min equ dis"]
        video_leng = data['Video length']
    if equiv_distance > should_walk:
        return ("Congratulations! You passed the test! ")
    else:
        return ("You failed the test. You can do better next time!")

    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        app.run(main_normal)
    except SystemExit:
        pass
