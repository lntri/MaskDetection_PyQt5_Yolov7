from libs import *


image_capture_name = ''


class MaskDetection(QMainWindow):
    def __init__(self):
        super(MaskDetection, self).__init__()
        self.thread_capture = None
        self.thread_realtime = None
        uic.loadUi('GUI_MaskDetection.ui', self)
        self.dirpath = QDir.currentPath()
        self.player = QtMultimedia.QMediaPlayer(None, QtMultimedia.QMediaPlayer.VideoSurface)
        self.cudnn_version = str(gpu_info()[0])
        self.cuda_device_name = gpu_info()[2]
        self.cuda_capacity = str(round(gpu_info()[3]))

        # Image
        self.width = 1299
        self.height = 589
        load_image(default_image, self.lblDisplayImageOutput, self.width, self.height)
        self.btnBrowseImage.clicked.connect(self.browse_image)

        self.btnDetectImage.clicked.connect(self.thread_image)

        self.lblCuDNNVersionImage.setText(self.cudnn_version)
        self.lblCUDADeviceNameImage.setText(self.cuda_device_name)
        self.lblCUDACapacityImage.setText(self.cuda_capacity)

        # Video
        # path_output = video_dir + 'pred_mask.mp4'
        # load_video(path_output, self.player, self.widgetVideoOutput)
        self.btnBrowseVideo.clicked.connect(self.browse_video)
        self.btnDetectVideo.clicked.connect(self.thread_video)
        self.btnViewResult.clicked.connect(self.load_result_video)
        self.btnResetVideo.clicked.connect(self.reset_video)
        self.lblCuDNNVersionVideo.setText(self.cudnn_version)
        self.lblCUDADeviceNameVideo.setText(self.cuda_device_name)
        self.lblCUDACapacityVideo.setText(self.cuda_capacity)

        # Take photo
        self.wc_width = 1299
        self.wc_height = 599
        load_image(default_image, self.lblDisplayWebcam, self.wc_width, self.wc_height)
        self.btnActiveWebcam.clicked.connect(self.active_webcam)
        self.btnDeactiveWebcam.clicked.connect(self.deactive_webcam)
        self.btnCapture.clicked.connect(self.capture)
        self.btnDetectTakePhoto.clicked.connect(self.detect_image_capture)

        # Real-time
        self.rt_width = 1299
        self.rt_height = 639
        load_image(default_image, self.lblDisplayRealtime, self.rt_width, self.rt_height)
        self.btnActiveRealtime.clicked.connect(self.active_realtime)
        self.btnDeactiveRealtime.clicked.connect(self.deactive_realtime)

    # ============== WEBCAM/CAMERA REAL-TIME PROCESSING ==============
    def active_realtime(self):
        self.btnActiveRealtime.setEnabled(False)
        self.btnDeactiveRealtime.setEnabled(True)
        self.lblDisplayRealtime.resize(self.rt_width, self.rt_height)
        self.thread_realtime = RealTimeThread()
        self.thread_realtime.change_pixmap_signal.connect(self.update_image_realtime)
        self.thread_realtime.start()

    def deactive_realtime(self):
        self.thread_realtime.stop()
        self.thread_realtime.change_pixmap_signal.disconnect()
        self.btnActiveRealtime.setEnabled(True)
        self.btnDeactiveRealtime.setEnabled(False)

    @pyqtSlot(np.ndarray)
    def update_image_realtime(self, cv_img):
        """Updates the lblDisplayWebcam with a new opencv image"""
        qt_img = convert_cv_qt(cv_img)
        load_image(qt_img, self.lblDisplayRealtime, self.rt_width, self.rt_height)

    # ============== WEBCAM/CAMERA PROCESSING ==============
    def active_webcam(self):
        self.btnActiveWebcam.setEnabled(False)
        self.btnDeactiveWebcam.setEnabled(True)
        self.btnCapture.setEnabled(True)
        self.btnDetectTakePhoto.setEnabled(False)
        self.lblDisplayWebcam.resize(self.wc_width, self.wc_height)
        self.thread_capture = WebcamThread()
        self.thread_capture.change_pixmap_signal.connect(self.update_image_capture)
        self.thread_capture.start()

    def deactive_webcam(self):
        self.thread_capture.stop()
        self.thread_capture.change_pixmap_signal.disconnect()
        self.btnActiveWebcam.setEnabled(True)
        self.btnDeactiveWebcam.setEnabled(False)
        self.btnCapture.setEnabled(False)

    def capture(self):
        self.thread_capture.stop()
        self.thread_capture.change_pixmap_signal.disconnect()
        image = ImageQt.fromqpixmap(self.lblDisplayWebcam.pixmap())
        self.btnActiveWebcam.setEnabled(True)
        self.btnDetectTakePhoto.setEnabled(True)
        self.btnDeactiveWebcam.setEnabled(False)
        self.btnCapture.setEnabled(False)
        global image_capture_name
        image_capture_name = 'capture_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.jpg'
        image.save(image_dir + image_capture_name)

    def detect_image_capture(self):
        # Input
        path_input = image_dir + image_capture_name

        # Detect
        detected_image = detect_single_image(path_input)
        detected_image_name = 'pred_' + image_capture_name
        path_output = image_dir + detected_image_name
        cv2.imwrite(path_output, detected_image)

        # Output
        load_image(path_output, self.lblDisplayWebcam, self.wc_width, self.wc_height)

    @pyqtSlot(np.ndarray)
    def update_image_capture(self, cv_img):
        """Updates the lblDisplayWebcam with a new opencv image"""
        qt_img = convert_cv_qt(cv_img)
        load_image(qt_img, self.lblDisplayWebcam, self.wc_width, self.wc_height)

    # ============== VIDEO PROCESSING ==============
    def browse_video(self):
        options = QFileDialog.Options()
        path_input, _ = QFileDialog.getOpenFileName(self, "Choose Video File...", "",
                                                    "Video files (*.mp4 *.avi)", options=options)
        if path_input != '':
            self.btnDetectVideo.setEnabled(True)
            self.btnResetVideo.setEnabled(True)
            self.txtPathVideo.setText(path_input)
        else:
            return

    def reset_video(self):
        self.btnDetectVideo.setEnabled(False)
        self.btnResetVideo.setEnabled(False)
        self.btnViewResult.setEnabled(False)
        self.probarVideo.setValue(0)
        self.txtPathVideo.setText('')
        self.player = QtMultimedia.QMediaPlayer(None, QtMultimedia.QMediaPlayer.VideoSurface)

    def thread_video(self):
        t2 = Thread(target=self.detect_video)
        t2.start()

    def load_result_video(self):
        path_input = self.txtPathVideo.text()
        video_name = path_input.split('/')[-1]
        path_output = video_dir + 'pred_' + video_name
        load_video(path_output, self.player, self.widgetVideoOutput)

    def detect_video(self):
        path_input = self.txtPathVideo.text()
        # Input
        video_name = 'pred_' + path_input.split('/')[-1]
        path_output = video_dir + video_name

        # Initializing video object
        video = cv2.VideoCapture(path_input)

        # Video information
        fps = video.get(cv2.CAP_PROP_FPS)
        w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        nframes = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        # Initialzing object for writing video output
        output = cv2.VideoWriter(path_output, cv2.VideoWriter_fourcc(*'DIVX'), fps, (w, h))
        torch.cuda.empty_cache()

        # Initializing model and setting it for inference
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

            classes = None
            if opt['classes']:
                classes = []
                for class_name in opt['classes']:
                    classes.append(opt['classes'].index(class_name))

            # self.probarVideo.setMaximum(nframes)
            for j in range(nframes):
                ret, img0 = video.read()
                if ret:
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

                    pred = non_max_suppression(pred, opt['conf-thres'], opt['iou-thres'], classes=classes,
                                               agnostic=False)
                    t2 = time_synchronized()
                    for i, det in enumerate(pred):
                        s = ''
                        s += '%gx%g ' % img.shape[2:]  # print string
                        gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]
                        if len(det):
                            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

                            for c in det[:, -1].unique():
                                n = (det[:, -1] == c).sum()  # detections per class
                                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                            for *xyxy, conf, cls in reversed(det):
                                label = f'{names[int(cls)]} {conf:.2f}'
                                plot_one_box(xyxy, img0, label=label, color=colors[int(cls)], line_thickness=2)

                    print(f"{j + 1}/{nframes} frames processed")
                    value_passed = j + 1
                    percentage = ((value_passed - 0) / (nframes - 0)) * 100
                    self.probarVideo.setValue(round(percentage))
                    output.write(img0)
                else:
                    break
            else:
                result = 'Done!'
                self.btnViewResult.setEnabled(True)
                print(result)
        output.release()
        video.release()
        return

    # ============== IMAGE PROCESSING ==============
    def browse_image(self):
        options = QFileDialog.Options()
        path_input = QFileDialog.getOpenFileName(self, "Choose Image File...", "",
                                                 "Image files (*.jpg *.jpeg *.png *.gif)", options=options)[0]
        if path_input != '':
            # Input
            self.btnDetectImage.setEnabled(True)
            self.txtPathImage.setText(path_input)
        else:
            return

    def thread_image(self):
        t1 = Thread(target=self.detect_image)
        t1.start()

    def detect_image(self):
        # Input
        path_input = self.txtPathImage.text()
        # Detect
        detected_image = detect_single_image(path_input)
        image_name = path_input.split('/')[-1]
        path_output = image_dir + 'pred_' + image_name
        cv2.imwrite(path_output, detected_image)
        # Output
        load_image(path_output, self.lblDisplayImageOutput, self.width, self.height)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mask_detection = MaskDetection()
    mask_detection.setWindowFlags(Qt.WindowMinimizeButtonHint | Qt.WindowCloseButtonHint)
    mask_detection.setFixedSize(1366, 768)
    mask_detection.show()
    app.exec_()
