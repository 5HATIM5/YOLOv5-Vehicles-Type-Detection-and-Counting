from flask import request, render_template, redirect, url_for, flash, jsonify, session, Response
from werkzeug.utils import secure_filename
import json
import numpy as np
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


@app.route('/')
def index():
    return render_template('index.html', status=session)


@torch.no_grad()
@app.route('/detect_all', methods=['GET', 'POST'])
def detect_all():
    if request.method == 'POST':
        source = request.json['source']
        print(source)

        # Make Directories
        save_dir = increment_path(os.path.join(cfg.OUTPUT_PATH, cfg.EXP_NAME))  # increment run
        (save_dir / 'crops' if cfg.SAVE_CROPS else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        vhl = cfg.Vehicle()
        print(cfg.SOURCE, cfg.IMGSIZE, cfg.STRIDE)
        im, im0s, path = LoadImage(cfg.SOURCE, img_size=cfg.IMGSIZE, stride=cfg.STRIDE, auto=True)

        # initialize important variables
        vehicles_counter_area = 0
        vehicles_counter = {name:0 for name in cfg.CLASSES}
        vehicles_counter_display = {name:0 for name in cfg.CLASSES}
        current_time_instance = datetime.now()

        processed_ids = []
        tracked_ids = dict()

        im = torch.from_numpy(im).to(cfg.DEVICE)
        im = im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        im = im[None] if len(im.shape) == 3 else im  # expand for batch dim

        # # draw area rectangle mask
        im0 = overlay(cfg, im0s.copy(), alpha=0.2, color=(0, 255, 0))

        # ROI y positions
        area_y1, area_y2 = im0s.shape[0] * cfg.AREA_THRESHOLD[0], im0s.shape[0] * cfg.AREA_THRESHOLD[1]

        # Inference
        pred = cfg.YOLO_MODEL(im, augment=vhl.augment, visualize=False)
        # NMS
        pred = non_max_suppression(pred, vhl.conf_thres, vhl.iou_thres, vhl.classes, vhl.agnostic_nms, max_det=vhl.max_det)

        # frame by frame so take the first one
        det = pred[0]
        # Process predictions
        bboxes, scores, labels = [], [], []
        imc = im0s.copy()

        p = Path(path)  # to Path
        save_path = str(save_dir / p.name)  # img.jpg
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
                    bboxes.append(xyxy)
                    scores.append(conf)
                    labels.append(cfg.CLASSES[int(cls)])

                # # annotate vehicle labels within area
                # label_ = f'{names[int(cls)]}'
                # annotator.box_label([x1, y1, x2, y2], label_, color=colors(cls, True))            

            # format bounding boxes from normalized xmin, ymin, xmax, ymax ---> xmin, ymin, width, height
            bboxes = format_boxes(bboxes)
            # encode yolo detections and feed to tracker
            features = cfg.DEEPSORT_ENCODER(im0, bboxes)
            detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, labels, features)]
            # Call the tracker
            cfg.DEEPSORT_TRACKER.predict()
            cfg.DEEPSORT_TRACKER.update(detections)

            # update tracks
            vehicles_counter_area = 0
            for track in cfg.DEEPSORT_TRACKER.tracks:
                # if not track.is_confirmed() or track.time_since_update > 1:
                #     continue
                
                # increament vehicle count within the area of ROI
                vehicles_counter_area += 1
                # get tracked object ids and boxes
                bbox, label, track_id = track.to_tlbr(), track.get_class(), track.get_id()

                # annotate vehicle labels within area
                label_ = f'{track_id}: {label}'
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                annotator.box_label([x1, y1, x2, y2], label_, color=colors(cls, True))            

                # continue if a vehicle value has been entered in record already
                if track_id in processed_ids:
                    continue

                # initialize tracking vehicle if it's the first time 
                if track_id not in tracked_ids:
                    tracked_ids[track_id] = {
                        "types": [],
                        "appearance": 0
                    }

                # save vehicle information against specific tracking id
                tracked_ids[track_id]["types"].append(label)
                tracked_ids[track_id]["appearance"] += 1

                # check if a certain vehicle has appeared more then 5 frames, then add it to the record
                if tracked_ids[track_id]["appearance"] > cfg.TRACKING_FRAMES_NUM:
                    
                    types = [t for t in tracked_ids[track_id]["types"] if t != '']
                    type_to_show = max(set(types), key = types.count) if len(types) > 0 else 'n/a'
                    vehicles_counter[type_to_show] += 1
                    vehicles_counter_display[type_to_show] += 1

                    # once the record is entered for specific tracking id, then don't process it again
                    del tracked_ids[track_id]
                    processed_ids.append(track_id)

        # draw results on frame
        im0 = draw_results(im0, vehicles_counter_display, vehicles_counter_area)
        im0 = annotator.result()

        # Save results (image with detections)
        if cfg.SAVE_IMG:
            cv2.imwrite(save_path, im0)

        if cfg.VIS:
            # resize the image for better visualization
            im0 = resizeImage(im0, width=1440)
            cv2.imshow("detection", im0)
            cv2.waitKey(0)

        print(f'Done...')
        cv2.destroyAllWindows()

    return jsonify({"status": True})