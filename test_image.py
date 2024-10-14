from PyQt5 import QtWidgets, QtGui
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QApplication, QFileDialog
import sys
import cv2
import imutils
import torch
import function.utils_rotate as utils_rotate
import function.helper as helper
import os
from gtts import gTTS
from playsound import playsound
import math
import numpy as np
import datetime
from layout_image import Ui_Frame

class MainWindow(QtWidgets.QFrame, Ui_Frame):
    def __init__(self,*args, **kwargs):
        super(MainWindow,self).__init__(*args, **kwargs)
        self.setupUi(self)
        self.btn_chonanh.clicked.connect(self.loadImage)
        self.btn_nhandang.clicked.connect(self.imgae_license)
        self.btn_amthanh.clicked.connect(self.on_amthanh_clicked)

        # Khởi tạo mô hình YOLO
        self.yolo_LP_detect = torch.hub.load('yolov5', 'custom', path='model/LP_detector.pt', force_reload=True, source='local')
        self.yolo_license_plate = torch.hub.load('yolov5', 'custom', path='model/LP_ocr.pt', force_reload=True, source='local')
        self.yolo_license_plate.conf = 0.60

    def showtime(self):
        while True:
            QApplication.processEvents()
            dt = datetime.datetime.now()
            self.let_ngay.setText('%s-%s-%s' % (dt.day, dt.month, dt.year))
            self.let_gio.setText('%s:%s:%s' % (dt.hour, dt.minute, dt.second))

    def loadImage(self):
        self.img_path = QFileDialog.getOpenFileName(filter="Image (*.*)")[0]
        self.Ivehicle = cv2.imread(self.img_path)

        # Nguyên gốc
        self.cv2_path = cv2.imread(self.img_path)

        self.img_goc = cv2.imread(self.img_path)
        self.setPhoto()

    def setPhoto(self):
        self.Ivehicle = imutils.resize(self.Ivehicle,width=300,height=340)
        frame = cv2.cvtColor(self.Ivehicle, cv2.COLOR_BGR2RGB)
        self.Ivehicle = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
        self.original_img.setPixmap(QtGui.QPixmap.fromImage(self.Ivehicle))

    def imgae_license(self):
        img = cv2.imread(self.img_path)
        plates = self.yolo_LP_detect(img, size=640)
        list_plates = plates.pandas().xyxy[0].values.tolist()
        list_read_plates = set()
        
        if len(list_plates) == 0:
            lp = helper.read_plate(self.yolo_license_plate, img)
            if lp != "unknown":
                self.display_plate(lp, img)
                list_read_plates.add(lp)
                self.info(lp)
        else:
            for plate in list_plates:
                flag = 0
                x = int(plate[0])  # xmin
                y = int(plate[1])  # ymin
                w = int(plate[2] - plate[0])  # xmax - xmin
                h = int(plate[3] - plate[1])  # ymax - ymin  
                crop_img = img[y:y+h, x:x+w]
                cv2.rectangle(img, (int(plate[0]),int(plate[1])), (int(plate[2]),int(plate[3])), color = (0,0,225), thickness = 2)
                for cc in range(0,2):
                    for ct in range(0,2):
                        lp = helper.read_plate(self.yolo_license_plate, utils_rotate.deskew(crop_img, cc, ct))
                        if lp != "unknown":
                            list_read_plates.add(lp)
                            self.display_plate(lp, img)
                            self.info(lp)
                            flag = 1
                            break
                    if flag == 1:
                        break
        self.show_result(img)
    

    def display_plate(self, plate, img):
        cv2.putText(img, plate, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        self.let_bienso.setText(plate)

    def show_result(self, img):
        self.img_goc = imutils.resize(img, width=971, height=541)
        frame = cv2.cvtColor(self.img_goc, cv2.COLOR_BGR2RGB)
        self.img_goc = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
        self.lbl_result.setPixmap(QtGui.QPixmap.fromImage(self.img_goc))

    def info(self, text): #nhận biển số xe từ người dùng, lấy hai ký tự đầu tiên của biển số để xác định tỉnh thành, 
                             #và hiển thị tỉnh thành tương ứng
        in4 = self.let_bienso.text()
        in5 = int(''.join(filter(str.isdigit, in4[0:2])))

        
        lang = {
            11: 'Cao Bằng', 12: 'Lạng Sơn', 14: 'Quảng Ninh', 15: 'Hải Phòng', 17: 'Thái Bình', 18: 'Nam Định',
            19: 'Phú Thọ', 20: 'Thái Nguyên', 21: 'Yên Bái', 22: 'Tuyên Quang', 23: 'Hà Giang', 24: 'Lao Cai',
            25: 'Lai Châu', 26: 'Sơn La', 27: 'Điện Biên', 28: 'Hoà Bình', 29: 'Hà Nội', 30: 'Hà Nội', 31: 'Hà Nội',
            32: 'Hà Nội', 33: 'Hà Nội', 40: 'Hà Nội', 34: 'Hải Dương', 35: 'Ninh Bình', 36: 'Thanh Hóa', 37: 'Nghệ An',
            38: 'Hà Tĩnh', 43: 'Đà Nẵng', 47: 'Dak Lak', 48: 'Đắc Nông', 49: 'Lâm Đồng', 50: 'Hồ chí Minh', 51: 'Hồ chí Minh',
            52: 'Hồ chí Minh',
            53: 'Hồ chí Minh', 54: 'Hồ chí Minh', 55: 'Hồ chí Minh', 56: 'Hồ chí Minh', 57: 'Hồ chí Minh', 58: 'Hồ chí Minh', 59: 'Hồ chí Minh', 60: 'Đồng Nai',
            61: 'Bình Dương',
            62: 'Long An', 63: 'Tiền Giang', 64: 'Vĩnh Long', 65: 'Cần Thơ', 66: 'Đồng Tháp', 67: 'An Giang',
            68: 'Kiên Giang',
            69: 'Cà Mau', 70: 'Tây Ninh', 71: 'Bến Tre', 72: 'Vũng Tàu', 73: 'Quảng Bình', 74: 'Quảng Trị', 75: 'Huế',
            76: 'Quảng Ngãi', 77: 'Bình Định', 78: 'Phú Yên', 79: 'Nha Trang', 81: 'Gia Lai', 82: 'Kon Tum',
            83: 'Sóc Trăng',
            84: 'Trà Vinh', 85: 'Ninh Thuận', 86: 'Bình Thuận', 88: 'Vĩnh Phúc', 89: 'Hưng Yên', 90: 'Hà Nam',
            92: 'Quảng Nam',
            93: 'Bình Phước', 94: 'Bạc Liêu', 95: 'Hậu Giang', 97: 'Bắc Cạn', 98: 'Bắc Giang', 99: 'Bắc Ninh',
        }

        for name, code in lang.items():
            if in5 == name:
                self.let_tinh.setText(code)
                print(f"Tỉnh: {code}")

    def on_amthanh_clicked(self):
        in4 = self.let_bienso.text()
        self.read_let_am_thanh(in4)

    def read_let_am_thanh(self, let_am_thanh):
        chu_cai_va_so = {
            'A': 'A', 'B': 'Bê', 'C': 'Xê', 'D': 'Đê', 'E': 'E', 'F': 'Ép', 'G': 'Gờ',
            'H': 'Hát', 'J': 'Giê', 'K': 'Ca', 'L': 'Lờ', 'M': 'Mờ', 'N': 'Nờ',
            'P': 'Pê', 'R': 'Rờ', 'S': 'Ét', 'T': 'Tê', 'V': 'Vê', 'W': 'Vê Kép',
            'X': 'Ích', 'Y': 'I', 'Z': 'Dét', '0': 'Không', '1': 'Một', '2': 'Hai', 
            '3': 'Ba', '4': 'Bốn', '5': 'Năm','6': 'Sáu', '7': 'Bảy', '8': 'Tám', '9': 'Chín'
        }
        chuoi_phat_am = " ".join([chu_cai_va_so.get(ky_tu.upper(), ky_tu) for ky_tu in let_am_thanh])

        tts = gTTS(f"Biển số xe là {chuoi_phat_am}", lang='vi')
        tts.save("bien_so_xe.mp3")
        playsound("bien_so_xe.mp3")
        os.remove("bien_so_xe.mp3")

if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    widget = MainWindow()
    widget.show()
    widget.showtime()
    try:
        sys.exit(app.exec_())
    except (SystemError, SystemExit):
        app.exit()
