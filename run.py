# import argparse
import torch
from utils.torch_utils import select_device
from utils.general import check_img_size
from models.common import DetectMultiBackend

# import deepsort
from deep_sort.tracker import Tracker
from deep_sort import nn_matching, generate_detections as gdet

import CONFIG as cfg
from app import app


@torch.no_grad()
def init():

    vhl = cfg.Vehicle()

    # initialize deep sort
    max_cosine_distance = 0.4
    nn_budget = None
    cfg.DEEPSORT_ENCODER = gdet.create_box_encoder(cfg.DEEPSORT_WEIGHTS_PATH, batch_size=1)
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # initialize tracker
    # tracker = Tracker(metric)
    cfg.DEEPSORT_TRACKER = Tracker(metric)

    # Initialize and Load car detection model
    cfg.DEVICE = select_device()
    model = DetectMultiBackend(vhl.weights, device=cfg.DEVICE, dnn=False)
    stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
    imgsz = check_img_size(vhl.imgsz, s=stride)  # check image size
    model.model.float()
    model(torch.zeros(1, 3, *imgsz).to(cfg.DEVICE).type_as(next(model.model.parameters())))  # warmup
    cfg.YOLO_MODEL = model

    cfg.CLASSES = names
    cfg.IMGSIZE = imgsz
    cfg.STRIDE = stride


if __name__ == "__main__":

    # parser = argparse.ArgumentParser()

    # Program input flags
    # These flags can be set in the command line interface when running the command
    # sample command is 
    # $ python run.py --flag1=value --flag2=value ...

    # parser.add_argument('--threshold', type=float, required=False, default=0.6,
    #     help='Probability threshold above which a person is to be classified, else unknown, default=%(default)s')

    # parser.add_argument('--training_images', type=int, required=False, default=10, 
    #     help='# of images required to train the classifier on, greater number brings better accuracy, default=%(default)s')

    # parser.add_argument('--gpu', default=True, action="store_true",
    #     help = "Specify whether to use GPU, default=%(default)s")

    # parser.add_argument('--resize_scale', type=float, required=False, default=1.0,
    #     help = "Resize the input video, e.g. 0.75 will resize to 1/4th of the video. default is 1 means No resize")
    

    # # parsing the flags into a variable called args
    # args = parser.parse_args()

    # CONFIG.TRAINING_IMAGES = args.training_images
    # CONFIG.CONFIDENCE_THRESHOLD = args.threshold
    # CONFIG.GPU = args.gpu
    # CONFIG.RESIZE_SCALE = args.resize_scale

    # init
    init()
    print("----------------- RUN -----------------------")

    # run the server
    app.run(host='0.0.0.0', debug=False)
