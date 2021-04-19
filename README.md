# yolov4-deepsort
[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)

The project is an website designed for walking distance and speed analysis of patients performing "Six Minute Walking Test". The web technical stack include Flask, PostgreSQL, Docker and Plotly. This Object tracking model is implemented with YOLOv4, DeepSort, and TensorFlow. 

Example video showing tracking of a Six-Minute Walking Test Patient:
<p align="center"><img src="./app/yolov4_deepsort/data/helpers/6mwt_demo.gif"\></p>

## Deploy with Docker
    ```sh
    docker build -t [image name] .
    docker run -d --name [container name] -p [host port:container port] [image name]
    ```

## Downloading Official YOLOv4 Pre-trained Weights
The object tracker uses YOLOv4 to make the object detections, which deep sort then uses to track. There exists an official pre-trained YOLOv4 object detector model that is able to detect 80 classes. For easy demo purposes we will use the pre-trained weights for our tracker.
Download pre-trained yolov4.weights file: https://drive.google.com/open?id=1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT

Copy and paste yolov4.weights from your downloads folder into the 'data' folder of this repository.

If you want to use yolov4-tiny.weights, a smaller model that is faster at running detections but less accurate, download file here: https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights

## Running the Tracker with YOLOv4
To implement the object tracking using YOLOv4, first we convert the .weights into the corresponding TensorFlow model which will be saved to a checkpoints folder. Then all we need to do is run the object_tracker.py script to run our object tracker with YOLOv4, DeepSort and TensorFlow.
<!-- ```bash
# Convert darknet weights to tensorflow model
python save_model.py --model yolov4 

# Run yolov4 deep sort object tracker on video
python object_tracker.py --video ./data/video/test.mp4 --output ./outputs/demo.avi --model yolov4

# Run yolov4 deep sort object tracker on webcam (set video flag to 0)
python object_tracker.py --video 0 --output ./outputs/webcam.avi --model yolov4
```
The output flag allows you to save the resulting video of the object tracker running so that you can view it again later. Video will be saved to the path that you set. (outputs folder is where it will be if you run the above command!)

If you want to run yolov3 set the model flag to ``--model yolov3``, upload the yolov3.weights to the 'data' folder and adjust the weights flag in above commands. (see all the available command line flags and descriptions of them in a below section) -->

<!-- # Running the Tracker with YOLOv4-Tiny
The following commands will allow you to run yolov4-tiny model. Yolov4-tiny allows you to obtain a higher speed (FPS) for the tracker at a slight cost to accuracy. Make sure that you have downloaded the tiny weights file and added it to the 'data' folder in order for commands to work!
```
# save yolov4-tiny model
python save_model.py --weights ./data/yolov4-tiny.weights --output ./checkpoints/yolov4-tiny-416 --model yolov4 --tiny

# Run yolov4-tiny object tracker
python object_tracker.py --weights ./checkpoints/yolov4-tiny-416 --model yolov4 --video ./data/video/test.mp4 --output ./outputs/tiny.avi --tiny
```

Example video showing tracking of all coco dataset classes:
<p align="center"><img src="data/helpers/all_classes.gif"\></p>

## Filter Classes that are Tracked by Object Tracker
By default the code is setup to track all 80 or so classes from the coco dataset, which is what the pre-trained YOLOv4 model is trained on. However, you can easily adjust a few lines of code in order to track any 1 or combination of the 80 classes. It is super easy to filter only the ``person`` class or only the ``car`` class which are most common.

To filter a custom selection of classes all you need to do is comment out line 159 and uncomment out line 162 of [object_tracker.py](https://github.com/theAIGuysCode/yolov4-deepsort/blob/master/object_tracker.py) Within the list ``allowed_classes`` just add whichever classes you want the tracker to track. The classes can be any of the 80 that the model is trained on, see which classes you can track in the file [data/classes/coco.names](https://github.com/theAIGuysCode/yolov4-deepsort/blob/master/data/classes/coco.names)

### References  
  * [tensorflow-yolov4-tflite](https://github.com/hunglc007/tensorflow-yolov4-tflite)
  * [Deep SORT Repository](https://github.com/nwojke/deep_sort) -->
