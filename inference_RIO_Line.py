from utils.augmentations import letterbox
from utils.torch_utils import select_device
from utils.plots import Annotator, colors
from utils.general import (
    check_img_size, increment_path, non_max_suppression, scale_coords, resizeImage, overlay, draw_results)
from utils.datasets import LoadImages, LoadStreams
from models.common import DetectMultiBackend
from pathlib import Path

import cv2
import torch

import os
import sys
import CONFIG as cfg
import numpy as np
import time
from tensorflow.keras.models import load_model
import tensorflow as tf
import pandas as pd
from datetime import datetime


@torch.no_grad()
def run(vhl):

    # press space to pause/play the video while inference
    stop = False

    # Directories
    save_dir = increment_path(os.path.join(cfg.OUTPUT_PATH, cfg.EXP_NAME))  # increment run
    (save_dir / 'crops' if cfg.SAVE_CROPS else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize and Load car detection model
    device = select_device(cfg.DEVICE)
    model = DetectMultiBackend(vhl.weights, device=device, dnn=False)
    stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
    imgsz = check_img_size(vhl.imgsz, s=stride)  # check image size
    model.model.float()
    model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup

    dataset = LoadImages(cfg.SOURCE, img_size=imgsz, stride=stride, auto=pt and not jit)
    bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    vehicles_counter_area = 0
    vehicles_counter = {name:0 for name in names}
    vehicles_counter_display = {name:0 for name in names}
    current_time_instance = datetime.now()
    df = pd.DataFrame(columns=[*['time instance'], *names])

    for path, im, im0s, vid_cap, s in dataset:
        # after 10 seconds of inference, update the current time
        if (datetime.now() - current_time_instance).total_seconds() > cfg.TIME_INSTANCE:
            # append the vehicle record to csv
            vehicles_counter['time instance'] = current_time_instance
            df = df.append(vehicles_counter, ignore_index=True)
            # export results to csv
            df.to_csv(os.path.join(save_dir, "result.csv"), index=False)
            current_time_instance = datetime.now()
            vehicles_counter = {name:0 for name in names}

        im = torch.from_numpy(im).to(device)
        im = im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        # draw area rectangle mask
        im0 = overlay(cfg, im0s.copy(), alpha=0.2, color=(204, 204, 0))

        # ROI y positions
        area_y1, area_y2 = int(im0s.shape[0] * cfg.AREA_THRESHOLD[0]), int(im0s.shape[0] * cfg.AREA_THRESHOLD[1])
        area_yc = (area_y1 + area_y2) // 2

        # draw ROI line in the middle of ROI area
        roi_pixels = cfg.LINE_OFFSET // 2
        cv2.rectangle(im0, (0, area_yc - roi_pixels), (im0.shape[1], area_yc + roi_pixels), (255, 255, 0), -1)
        
        # Inference
        pred = model(im, augment=vhl.augment, visualize=False)
        # NMS
        pred = non_max_suppression(pred, vhl.conf_thres, vhl.iou_thres, vhl.classes, vhl.agnostic_nms, max_det=vhl.max_det)
        # frame by frame so take the first one
        det = pred[0]
        imc = im0s.copy()

        p = Path(path)  # to Path
        save_path = str(save_dir / p.name)  # img.jpg
        annotator = Annotator(im0, line_width=vhl.line_thickness, example=str(names))

        # continue to the next frame if there's no detections
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

            # Write results
            vehicles_counter_area = 0   # reset the count within the are to zero
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)
                label = names[c]
                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                # counting vehicles
                center = (int((x1 + x2) // 2), int((y1 + y2) // 2))
                cv2.circle(im0, center, 5, (0, 0, 255), -1)     # draw the center of detection
                
                # if center of a detection box touched the ROI line then increment individual count            
                # save vehicle count against specific class
                if area_yc - roi_pixels <= center[1] <= area_yc + roi_pixels:
                    cv2.rectangle(im0, (0, area_yc - cfg.LINE_OFFSET), (im0.shape[1], area_yc + cfg.LINE_OFFSET), (255, 0, 0), -1)
                    vehicles_counter[label] += 1
                    vehicles_counter_display[label] += 1
                    
                # if center of detection box lies within the area then increament by one                
                if area_y1 <= center[1] <= area_y2:
                    vehicles_counter_area += 1

                # annotate vehicle labels within area
                annotator.box_label([x1, y1, x2, y2], label, color=colors(c, True))
                
                # saving crops
                if cfg.SAVE_CROPS:
                    imgVeh = imc[y1:y2, x1:x2]
                    crops_path = save_dir / 'crops'
                    crop_path = f"{crops_path}/{p.name[:-4]}_{datetime.now().strftime('%y%m%d_%H%M%S')}.jpg"
                    cv2.imwrite(crop_path, imgVeh)

        # draw results on frame
        im0 = draw_results(im0, vehicles_counter_display, vehicles_counter_area)
        im0 = annotator.result()

        # Save results (image with detections)
        if cfg.SAVE_IMG:
            if dataset.mode == 'image':
                cv2.imwrite(save_path, im0)
            else:  # 'video' or 'stream'
                if vid_path[0] != save_path:  # new video
                    vid_path[0] = save_path
                    if isinstance(vid_writer[0], cv2.VideoWriter):
                        vid_writer[0].release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path += '.mp4'
                    vid_writer[0] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer[0].write(im0)

        if cfg.VIS:
            # resize the image for better visualization
            im0 = resizeImage(im0, width=1440)
            cv2.imshow("detection", im0)
            if cv2.waitKey(0 if stop else 1) == 32:
                stop = not stop

    print(f'Done...')
    cv2.destroyAllWindows()


if __name__ == "__main__":
    vhl = cfg.Vehicle()
    run(vhl)
