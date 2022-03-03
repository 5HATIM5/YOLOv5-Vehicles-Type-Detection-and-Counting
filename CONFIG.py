import os

""" Genreal Parameters """

SOURCE = "test_videos/test_image.jpg"
# SOURCE = "test_image"

VIS = True
SAVE_CROPS = False   # save cropped prediction boxes
SAVE_IMG = True      # save images/video 

OUTPUT_PATH = 'runs/detect'  # save results to project/name
EXP_NAME = 'exp'  # save results to project/name

# the percentage of height of frame within which the models will make predictions
# AREA_THRESHOLD is defined so the models do not process the entire frame, but instead, a part of frame (ROI)
# this brings significant improvements in speed
AREA_THRESHOLD = [0.55, 0.9] # [start_y, end_y]
assert AREA_THRESHOLD[0] < AREA_THRESHOLD[1], "AREA_THRESHOLD: start should be less then the end"

# number of seconds representing time instance to save records in csv
TIME_INSTANCE = 5


""" ROI Line Script Parameters """
# number of pixels representing line width used to increament or decreament vehicles count
LINE_OFFSET = 4


""" Vehicles Tracking Script Parameters """
# Path to pre-trained DeepSort weights
DEEPSORT_WEIGHTS_PATH = 'weights/mars-small128.pb'
assert os.path.exists(DEEPSORT_WEIGHTS_PATH), f"{DEEPSORT_WEIGHTS_PATH} doesn't exist"

# in how many frames in which a certain objects has appeared will be considered?
TRACKING_FRAMES_NUM = 4

""" YOLOv5 Model Parameters """
class Vehicle:
    def __init__(self):
        # self.weights = "weights/car_detection_best.pt"
        self.weights = "weights/vehicles_detection_best.pt"
        self.imgsz = [608]  # inference size (pixels)
        self.imgsz *= 2 if len(self.imgsz) == 1 else 1

        self.conf_thres = 0.5  # confidence threshold
        self.iou_thres = 0.4  # NMS IOU threshold
        self.max_det = 1000  # maximum detections per image

        self.classes = None  # filter by class: --class 0, or --class 0 2 3
        self.agnostic_nms = False  # class-agnostic NMS
        
        self.augment = False  # augmented inference
        self.visualize = False  # visualize features
        self.update = False  # update all models
        
        
        self.exist_ok = False  # existing project/name ok, do not increment
        self.line_thickness = 2  # bounding box thickness (pixels)
        self.hide_labels = False  # hide labels
        self.hide_conf = False  # hide confidences  




######################### Flask Configurations #########################
DEEPSORT_TRACKER = None
DEEPSORT_ENCODER = None
YOLO_MODEL = None
DEVICE = None
CLASSES = None
IMGSIZE = None
STRIDE = None

TEMP_IMAGE_PATH = "app/static/temp/temp.jpg"