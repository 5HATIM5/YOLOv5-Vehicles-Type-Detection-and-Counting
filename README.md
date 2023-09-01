<div>
<h1>YOLOv5 - Vehicles Type Detection and Counting</h1>
<br>

<p>
<img width="100%" src="./gallery/detections_roi_line.png"></a>
</p>
<p>

This repository focuses on the detection and classification of vehicles types and counting using machine learning and computer vision techniques. Yolov5 has been used for the detection perposes in collaboration with deep sort model for object tracking.
</p>
</div>

## <div>General Capabilities</div>

This repository implements the following capabilities:

* <b>Detection and classification of the vehicles:</b> YOLOv5 small model was trained on 7 different classes (car, truck, bike, motorcycle, bus, padestrain, ambulance).

* <b>Vehicle Counting:</b>Vehicle Counting within the ROI area (specified in the CONFIG.py file) as well as the total count per class is implemented in this repository. Vehicles count was implemented in a two different ways. First by implementing vehicles tracking algorithm using DeepSort which assigns a unique id to every detection bounding box in sequencial frames. Second by specifying a ROI Line in which the count is increamented if the center of a detection box touched that line. See the resultant video in gallery directory.

* <b>Exporting Results:</b> The count against every vehicle type in specific time instance is exported in a CSV for every run. In addition, it exports the processed video or images.

<br>
<div align="center">
<img width="auto" src="./gallery/results.png"></a>
</div>
<br>

## <div>Quick Start Examples</div>

<details open>
<summary><h3><b>Install</b></h3></summary>

[**Python>=3.7.0**](https://www.python.org/) is required with all
requirements.txt installed including
[**PyTorch>=1.7**](https://pytorch.org/get-started/locally/):

Download the repository and cd into the root direcotry.
```bash
$ pip install -r requirements.txt
```

</details>

<details open>
<summary><h3><b>Inference</b></h3></summary>

Inference can be performed in two different ways. For vehicles tracking inference, run inference_tracking.py script and inference_roi_Line.py for ROI Line. But before we have to make sure the configurations are set in the CONFIG.py file, e.g. SOURCE is the path to any video or images directory or an image.

```bash
    python inference_tracking.py
```

</details>

<details open>
<summary><h3><b>Configuration File</b></h3></summary>
The project configuration has been all defined in the CONFIG.py script. Some of the configurations need to be set according to the source video or images but most of them remains constant. Following is the breif explaination for every parameter:

<br>
<h3>General Configurations</h3>

* <b>SOURCE: </b>Path to input image, video or directory containing images/videos for inference
* <b>VIS: </b>Visualize detections during inference if True
* <b>SAVE_CROPS: </b>Save cropped prediction boxes if True
* <b>OUTPUT_PATH: </b>Save path for crops and results.csv
* <b>TIME_INSTANCE: </b> Number of seconds representing time instance after which records are being saved in csv
* <b>TRACKING_FRAMES_NUM: </b> In how many frames in which a certain objects has appeared will be considered for counting in case of vehicles tracking.
* <b>LINE_OFFSET: </b> ROI Line thickness in number of pixels
* <b>AREA_THRESHOLD: </b>This is very important parameter which need to be adjusted according to video/image. Area threshold specifies Region of Interest (ROI) and is an array of two elements which is start and end in percentage respectively. It basically represents the y values of a frame within which the detections will be considered. For example, [0.55, 0.9] will mark area from 55% to 95% of the image height from top to bottom as shown below:

<br>
<p float="left">
  <img width="100%" src="./gallery/detections_tracking.png">
</p>
<br>

<br>
<h3>YOLO Configurations</h3>
    YOLOv5 model is working in this project and it has some specific parameters. Hence, a separate class has been created named Vehicle:

<br>

* <b>self.weights: </b>Path to weights file e.g. /weights/yolov5s_best.pt
* <b>imgsz: </b>model input image size on which it was trained
* <b>conf_thres: </b>confidence threshold used while NMS
* <b>iou_thres: </b>IoU threshold used while NMS
* <b>line_thickness: </b>used to scale the bounding boxes drawn on the frame

</details>

## <div>References</div>

* YOLOv5 https://github.com/ultralytics/yolov5
* Deep SORT 

    Paper: https://arxiv.org/abs/1703.07402

    Github: https://github.com/nwojke/deep_sort

* VEHICLE DETECTION, TRACKING AND COUNTING https://github.com/ahmetozlu/vehicle_counting_tensorflow
