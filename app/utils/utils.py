import cv2
from pathlib import Path
from utils.augmentations import Albumentations, augment_hsv, copy_paste, letterbox, mixup, random_perspective
import numpy as np

IMG_FORMATS = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes

def LoadImage(source, img_size=640, stride=32, auto=True):

    if type(source) == str:
        p = str(Path(source).resolve())  # os-agnostic absolute path
        if not p.split('.')[-1].lower() in IMG_FORMATS:
            raise Exception(f'ERROR: {p} is not a valid image')

        img0 = cv2.imread(source)  # BGR
    else:
        img0 = source

    assert img0 is not None, f'Image Not Found {img0}'

    # Padded resize
    img = letterbox(img0, img_size, stride=stride, auto=True)[0]

    # Convert
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)
    return img, img0
