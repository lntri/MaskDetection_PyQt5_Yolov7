import torch
import cv2
import numpy as np
import random
import io
import time
import sys
from utils.general import check_img_size, non_max_suppression, scale_coords, set_logging
from utils.torch_utils import select_device, time_synchronized
from models.experimental import attempt_load
from utils.plots import plot_one_box
from datetime import datetime
from base64 import b64encode
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog
from PyQt5.QtCore import QDir, QThread, pyqtSignal, pyqtSlot, Qt
from PyQt5.QtGui import QPixmap
from PyQt5 import QtMultimedia, QtCore, QtGui, uic
from PIL import ImageQt
from threading import Thread



image_dir = 'images/'
video_dir = 'videos/'
default_image = 'images/yolo-v7.jpg'


classes_to_filter = ['with_mask', 'without_mask']
opt = {
    "weights": "weights/best.pt",  # Path to weights file default weights are for nano model
    "yaml": "data/data.yaml",
    "img-size": 640,  # default image size
    "conf-thres": 0.25,  # confidence threshold for inference.
    "iou-thres": 0.45,  # NMS IoU threshold for inference.
    "device": '0',  # device to run our model i.e. 0 or 0,1,2,3 or cpu
    "classes": classes_to_filter  # list of classes to filter or None
}


def gpu_info():
    cudnn_version = ''
    number_devices = ''
    cuda_device_name = ''
    cuda_capacity = ''
    cuda = torch.cuda.is_available()
    if cuda:
        cudnn_version = torch.backends.cudnn.version()
        number_devices = torch.cuda.device_count()
        cuda_device_name = torch.cuda.get_device_name(0)
        cuda_capacity = torch.cuda.get_device_properties(0).total_memory / 1e9
    return cudnn_version, number_devices, cuda_device_name, cuda_capacity


def load_image(image_path, label_widget, width, height):
    pixmap = QtGui.QPixmap(image_path)
    if not pixmap.isNull():
        label_widget.clear()
        label_widget.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        label_widget.resize(width, height)
        if pixmap.width() < width and pixmap.height() < height:
            label_widget.setPixmap(pixmap)
        else:
            label_widget.setPixmap(pixmap.scaled(width, height, Qt.KeepAspectRatio, Qt.SmoothTransformation))


def load_video(video_path, multimedia_widget, video_widget):
    multimedia_widget.setMedia(QtMultimedia.QMediaContent(QtCore.QUrl.fromLocalFile(video_path)))
    multimedia_widget.setVideoOutput(video_widget)
    multimedia_widget.play()


def convert_cv_qt(cv_img):
    """Convert from an opencv image to QPixmap"""
    rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb_image.shape
    bytes_per_line = ch * w
    convert_to_qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
    # p = convert_to_qt_format.scaled(self.wc_width, self.wc_height, Qt.KeepAspectRatio)
    return QPixmap.fromImage(convert_to_qt_format)


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def detect_single_image(source_image_path):
    with torch.no_grad():
        weights, imgsz = opt['weights'], opt['img-size']
        set_logging()
        device = select_device(opt['device'])
        half = device.type != 'cpu'
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check img_size
        if half:
            model.half()

        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))

        img0 = cv2.imread(source_image_path)
        img = letterbox(img0, imgsz, stride=stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=False)[0]

        # Apply NMS
        classes = None
        if opt['classes']:
            classes = []
            for class_name in opt['classes']:
                classes.append(opt['classes'].index(class_name))

        pred = non_max_suppression(
            pred, opt['conf-thres'], opt['iou-thres'], classes=classes, agnostic=False)
        t2 = time_synchronized()
        for i, det in enumerate(pred):
            s = ''
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]
            if len(det):
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], img0.shape).round()

                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                for *xyxy, conf, cls in reversed(det):
                    label = f'{names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, img0, label=label,
                                 color=colors[int(cls)], line_thickness=2)
        return img0


class WebcamThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self._run_flag = True

    def run(self):
        # capture from web cam
        cap = cv2.VideoCapture(0)
        while self._run_flag:
            ret, cv_img = cap.read()
            if ret:
                self.change_pixmap_signal.emit(cv_img)
        # shut down capture system
        cap.release()

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()


class RealTimeThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self._run_flag = True

    def run(self):
        # capture from web cam
        cap = cv2.VideoCapture(0)
        with torch.no_grad():
            weights, imgsz = opt['weights'], (480, 640)
            set_logging()
            device = select_device(opt['device'])
            half = device.type != 'cpu'
            model = attempt_load(weights, map_location=device)  # load FP32 model
            stride = int(model.stride.max())  # model stride

            if half:
                model.half()

            names = model.module.names if hasattr(model, 'module') else model.names
            colors = [[random.randint(0, 255) for _ in range(2)] for _ in names]
            if device.type != 'cpu':
                model(torch.zeros(1, 3, imgsz[0], imgsz[1]).to(device).type_as(next(model.parameters())))
            classes = None
            if opt['classes']:
                classes = []
                for class_name in opt['classes']:
                    classes.append(opt['classes'].index(class_name))
            while self._run_flag:
                ret, cv_img = cap.read()
                if ret:
                    img = letterbox(cv_img, imgsz, stride=stride)[0]
                    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
                    img = np.ascontiguousarray(img)
                    img = torch.from_numpy(img).to(device)
                    img = img.half() if half else img.float()  # uint8 to fp16/32
                    img /= 255.0  # 0 - 255 to 0.0 - 1.0
                    if img.ndimension() == 3:
                        img = img.unsqueeze(0)

                    # Inference
                    t1 = time_synchronized()
                    pred = model(img, augment=False)[0]

                    # Apply NMS
                    pred = non_max_suppression(pred, opt['conf-thres'], opt['iou-thres'], classes=classes,
                                               agnostic=False)
                    t2 = time_synchronized()
                    for i, det in enumerate(pred):
                        s = ''
                        s += '%gx%g ' % img.shape[2:]  # print string
                        gn = torch.tensor(cv_img.shape)[[1, 0, 1, 0]]
                        if len(det):
                            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], cv_img.shape).round()

                            for c in det[:, -1].unique():
                                n = (det[:, -1] == c).sum()  # detections per class
                                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                            for *xyxy, conf, cls in reversed(det):
                                label = f'{names[int(cls)]} {conf:.2f}'
                                plot_one_box(xyxy, cv_img, label=label, color=colors[int(cls)], line_thickness=2)
                    self.change_pixmap_signal.emit(cv_img)
        # shut down capture system
        cap.release()

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()

