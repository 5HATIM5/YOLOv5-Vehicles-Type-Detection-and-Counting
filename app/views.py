from flask import request, render_template, redirect, url_for, flash, jsonify, session, Response
from werkzeug.utils import secure_filename
import json
import numpy as np
import base64
import os
from app import app

import cv2
import torch
from utils.general import (increment_path, non_max_suppression, scale_coords, resizeImage, overlay, draw_results)
from utils.plots import Annotator, colors
from pathlib import Path

from app.utils.utils import LoadImage
from datetime import datetime

import CONFIG as cfg

from deep_sort.detection import Detection
# from deep_sort import nn_matching, generate_detections as gdet
from deep_sort.utils import format_boxes



import re
def decode_base64(data, altchars=b'+/'):
    """Decode base64, padding being optional.

    :param data: Base64 data as an ASCII byte string
    :returns: The decoded byte string.

    """

    data = data.split(",")[-1].encode('ascii')
    data = re.sub(rb'[^a-zA-Z0-9%s]+' % altchars, b'', data)  # normalize
    missing_padding = len(data) % 4
    if missing_padding:
        data += b'='* (4 - missing_padding)
    return base64.b64decode(data, altchars=b'+/')


@app.route('/')
def index():
    return render_template('index.html', status=session)


@torch.no_grad()
@app.route('/count_vehicles', methods=['GET', 'POST'])
def detect_all():
    if request.method == 'POST':
        base64_code = request.json['source']
        request_class = request.json['class']
        area_threshold_top = float(request.json['area_threshold_top'])
        area_threshold_bottom = float(request.json['area_threshold_bottom'])

        if request_class not in cfg.CLASSES:
            request_class = "all"

        # save the temporary image
        decodeit = open(cfg.TEMP_IMAGE_PATH, 'wb')
        decodeit.write(decode_base64(base64_code))
        decodeit.close()

        vhl = cfg.Vehicle()
        im, im0s = LoadImage(cfg.TEMP_IMAGE_PATH, img_size=cfg.IMGSIZE, stride=cfg.STRIDE, auto=True)

        # initialize important variables
        vehicles_counter = {
            name:{
                "count": 0,
                "detections": []
            } for name in cfg.CLASSES
            }

        im = torch.from_numpy(im).to(cfg.DEVICE)
        im = im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        im = im[None] if len(im.shape) == 3 else im  # expand for batch dim

        # # draw area rectangle mask
        im0 = overlay(cfg, im0s.copy(), alpha=0.2, color=(0, 255, 0))

        # ROI y positions
        area_y1, area_y2 = im0s.shape[0] * area_threshold_top, im0s.shape[0] * area_threshold_bottom

        # Inference
        pred = cfg.YOLO_MODEL(im, augment=vhl.augment, visualize=False)
        # NMS
        pred = non_max_suppression(pred, vhl.conf_thres, vhl.iou_thres, vhl.classes, vhl.agnostic_nms, max_det=vhl.max_det)

        # frame by frame so take the first one
        det = pred[0]

        # p = Path(path)  # to Path
        # save_path = str(save_dir / p.name)  # img.jpg
        annotator = Annotator(im0, line_width=vhl.line_thickness, example=str(cfg.CLASSES))

        # if there're detections
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

            # add detections to the tracker
            for *xyxy, conf, cls in reversed(det):
                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                # counting vehicles
                center = (int((x1 + x2) // 2), int((y1 + y2) // 2))
                cv2.circle(im0, center, 4, (0, 0, 255), -1)
                
                # if a vehicle is within the area of interest, then condider the box for tracking
                if area_y1 <= center[1] <= area_y2 and 0 <= center[0] <= im0.shape[1]:
                    # get tracked object ids and boxes
                    label = cfg.CLASSES[int(cls)]
                    annotator.box_label([x1, y1, x2, y2], label, color=colors(cls, True))
                    vehicles_counter[label]["count"] += 1
                    vehicles_counter[label]["detections"].append({
                        "bbox": {
                            "xmin": str(x1),
                            "ymin": str(y1),
                            "xmax": str(x2),
                            "ymax": str(y2),
                        },
                        "confidence score": str(float(conf))
                    })

        # draw results on frame
        # im0 = draw_results(im0, vehicles_counter_display, vehicles_counter_area)
        im0 = annotator.result()

        retval, buffer = cv2.imencode('.jpg', im0)
        im_encode = base64.b64encode(buffer)
        vehicles_counter["media"] = im_encode.decode('utf-8')

        # if cfg.VIS:
        #     # resize the image for better visualization
        #     img_to_show = resizeImage(im0.copy(), width=1440)
        #     cv2.imshow("detection", img_to_show)
        #     cv2.waitKey(0)

        # print(f'Done...')
        # cv2.destroyAllWindows()

    return (
        jsonify(vehicles_counter) if request_class == "all" else jsonify({
            "media": vehicles_counter["media"],
            request_class: vehicles_counter[request_class]
        })
    )


@app.route('/test', methods=['GET', 'POST'])
def test():
    return jsonify({"status": True})