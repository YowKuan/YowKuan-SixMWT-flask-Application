from app.yolov4_deepsort.object_tracker import main_normal, compare_walk_dis
from flask import Flask,redirect,url_for, render_template, request, send_from_directory, make_response
from absl import app, flags, logging
from absl.flags import FLAGS
import os
from werkzeug.utils import secure_filename
import json
import pandas as pd
from flask import current_app as web
from datetime import datetime as dt
from app.model.Video import Video
from app import db

UPLOAD_FOLDER = './app/yolov4_deepsort/Uploaded_video'
ALLOWED_EXTENSIONS = set(['mp4', 'png', 'jpg', 'jpeg', 'gif', 'MOV'])
web.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
web.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 16MB

#Defined flags for yolov4-deepsort model
flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './app/yolov4_deepsort/checkpoints/yolov4-416','path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.05, 'iou threshold')
flags.DEFINE_float('score', 0.35, 'score threshold')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_boolean('info', False, 'show detailed info of tracked objects')
flags.DEFINE_boolean('count', False, 'count objects being tracked on screen') 
flags.DEFINE_integer('limited_time', 10, 'walk 10m within this time to pass')
flags.DEFINE_integer('desired_dis', 8, 'Set the desire distance the patient need to walk')

@web.route('/')
def home():
    return render_template('6mwt_homepage.html')

#Input Section
@web.route('/input_normal_mode', methods =['POST','GET'])
def input_normal_mode():
    return render_template('input_normal_mode.html')
@web.route('/input_time_mode', methods =['POST','GET'])
def input_time_mode():
    return render_template('input_time_mode.html')
@web.route('/input_realtime', methods =['POST','GET'])
def input_realtime():
    return render_template('input_realtime.html')

#Upload the file
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@web.route('/upload', methods=['GET', 'POST'])
def upload_file():
    return render_template('upload.html')

@web.route('/upload_timemode', methods=['GET', 'POST'])
def upload_file_time():
    return render_template('upload_timemode.html')

#run normal mode
@web.route('/normal_mode_results', methods =['POST','GET'])
def walk_result():
    if request.method == 'POST':
        with open('./app/yolov4_deepsort/walk_result.json', newline='') as jsonfile:
            data = json.load(jsonfile)
            walk_distance = data["Distance"]
            equiv_distance = data["6min equ dis"]
            video_leng = data['Video length']
        user_name = request.form.get('username')
        patient_h = int(request.values.get('pat_height'))
        patient_w = int(request.values.get('pat_weight'))
        patient_a = int(request.values.get('pat_age'))
        p_should_walk = round(7.57*patient_h -5.02*patient_a -1.76*patient_w -309, 2)
        equ_should_walk = int(p_should_walk*video_leng/360)
        equ_walk_dis = round(walk_distance*360/video_leng, 2)
        return render_template('results_normal_mode.html', pri=compare_walk_dis(patient_h, patient_w, patient_a), 
        user=user_name, have_walk = equ_walk_dis, should_walk = p_should_walk) 

@web.route('/realtime_results', methods =['POST','GET'])
def realtime_result():
    if request.method == 'POST':
        with open('walk_result.json', newline='') as jsonfile:
            data = json.load(jsonfile)
            walk_distance = data["Distance"]
            equiv_distance = data["6min equ dis"]
            video_leng = data['Video length']
        user_name = request.form.get('username')
        patient_h = int(request.values.get('pat_height'))
        patient_w = int(request.values.get('pat_weight'))
        patient_a = int(request.values.get('pat_age'))
        p_should_walk = round(7.57*patient_h -5.02*patient_a -1.76*patient_w -309, 2)
        equ_should_walk = int(p_should_walk*video_leng/360)
        equ_walk_dis = round(walk_distance*360/video_leng, 2)
        return render_template('results_realtime.html', pri=compare_walk_dis(patient_h, patient_w, patient_a), 
        user=user_name, have_walk = equ_walk_dis, should_walk = p_should_walk) 

    
@web.route('/time_mode_results', methods =['POST','GET'])
def timed_result():
    if request.method == 'POST':
        user = request.form.get('username')
        t_limit = int(request.values.get('time_limit'))
        d_dis = int(request.values.get('desire_dis'))
        return object_timed_result(d_dis)
    return render_template('results_normal_mode.html')

@web.route('/run_normal_mode', methods =['POST','GET'])
def exe():
    if request.method == 'POST':
        vid_file = request.files['file']
        if vid_file and allowed_file(vid_file.filename):
            filename = secure_filename(vid_file.filename)
            vid_file.save(os.path.join(UPLOAD_FOLDER, filename))
            print(vid_file.filename)
            vid_route = './app/yolov4_deepsort/Uploaded_video/'+str(vid_file.filename)
    vid_route_json = {}
    vid_route_json = {'Video route':vid_route}
    video_name = vid_file.filename.split('.')[0]
    Group = video_name.split('-')[0]
    ID = video_name.split('_')[0]
    Dt = video_name.split('_')[1]
    Date = (Dt[0:4]+'-'+Dt[4:6]+'-'+Dt[6:8])
    #Insert video name into database for checking duplicate video
    videoname_insert = Video(patient_id=video_name)
    db.session.add(videoname_insert)
    db.session.commit()

    with open('./app/yolov4_deepsort/vid_route.json', 'w') as routefile:
        json.dump(vid_route_json, routefile) 
    try:
        app.run(main_normal)
    except SystemExit:
        pass
    return redirect(url_for('input_normal_mode'))

@web.route('/run_timemode', methods =['POST','GET'])
def exe_timemode():
    if request.method == 'POST':
        vid_file = request.files['file']
        if vid_file and allowed_file(vid_file.filename):
            filename = secure_filename(vid_file.filename)
            vid_file.save(os.path.join(UPLOAD_FOLDER, filename))
            vid_route = './Uploaded_video/'+str(vid_file.filename)
    vid_route_json = {}
    vid_route_json = {'Video route':vid_route}

    with open('vid_route.json', 'w') as routefile:
        json.dump(vid_route_json, routefile)
    try:
        app.run(main_timemode)
    except SystemExit:
        pass

    return redirect(url_for('input_time_mode'))

@web.route('/run_realtime')
def exe_realtime():
    try:
        app.run(main_realtime)
    except SystemExit:
        pass

    return redirect(url_for('input_realtime'))