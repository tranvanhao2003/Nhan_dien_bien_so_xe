import sys
import cv2
import datetime
import imutils
import numpy as np
import torch
import os
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QFileDialog, QApplication
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer
from gtts import gTTS
from playsound import playsound

# Your custom imports (adjust paths as necessary)
import function.utils_rotate as utils_rotate
import function.helper as helper
import layout_video

class MainWindow(QtWidgets.QFrame, layout_video.Ui_Frame):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setupUi(self)
        
        # Kết nối các nút với chức năng tương ứng
        self.btn_chonVideo_2.clicked.connect(self.loadVideo)
        self.btn_chonVideo.clicked.connect(self.resumeVideoCapture)  # Thay đổi để tiếp tục video
        self.btn_nhandang.clicked.connect(self.pauseVideoCapture)  # Dừng video khi nhấn nút này
        self.thread = {}
        self.btn_amthanh.clicked.connect(self.on_amthanh_clicked)
      
        # Tải các mô hình YOLO
        self.yolo_LP_detect = torch.hub.load('yolov5', 'custom', path='model/LP_detector_nano_61', force_reload=True, source='local')
        self.yolo_license_plate = torch.hub.load('yolov5', 'custom', path='model/LP_ocr_nano_62', force_reload=True, source='local')
        self.yolo_license_plate.conf = 0.60

        self.video_thread = None  # Giữ luồng xử lý video
        self.img_path = None
        self.is_video_processing = False
        self.is_paused = False  # Theo dõi trạng thái tạm dừng

    def loadVideo(self):
        """Tải một tệp video và bắt đầu xử lý."""
        video_path = QFileDialog.getOpenFileName(filter="Video Files (*.mp4 *.avi *.mov)")[0]
        if video_path:
            self.video_path = video_path  # Lưu đường dẫn video để sử dụng sau
            print(f"Đã tải video: {self.video_path}")  # Để kiểm tra xem video đã được tải chưa

    def closeEvent(self, event):
        self.pauseVideoCapture()  # Dừng xử lý video khi đóng cửa sổ

    def pauseVideoCapture(self):
        """Tạm dừng xử lý video."""
        if self.video_thread is not None and self.video_thread.isRunning():
            self.video_thread.pause()  # Tạm dừng luồng video
            self.is_paused = True

    def resumeVideoCapture(self):
        """Tiếp tục xử lý video nếu đang tạm dừng, hoặc bắt đầu nếu chưa bắt đầu."""
        if self.video_path:  # Kiểm tra xem đã chọn video chưa
            if self.is_paused and self.video_thread is not None:
                # Tiếp tục nếu đang tạm dừng
                self.video_thread.resume()
                self.is_paused = False
            else:
                # Bắt đầu một luồng video mới nếu chưa tạm dừng
                if self.video_thread is not None and self.video_thread.isRunning():
                    self.video_thread.stop()

                self.video_thread = CaptureVideo(self.video_path, self.yolo_LP_detect, self.yolo_license_plate, self)
                self.video_thread.signal.connect(self.updateFrame)
                self.video_thread.updateBs.connect(self.let_bienso.setText)
                self.video_thread.start()
                self.is_video_processing = True
        else:
            print("Chưa chọn video. Vui lòng chọn video trước.")  # Đầu ra để kiểm tra

   
    def updateFrame(self, img):
        """Cập nhật khung hình hiển thị."""
        qt_img = self.convert_cv_qt(img)
        self.original_video.setPixmap(qt_img)
         # Hiển thị ngày giờ
        dt = datetime.datetime.now()
        self.let_ngay.setText('%s-%s-%s' % (dt.day, dt.month, dt.year))
        self.let_gio.setText('%s:%s:%s' % (dt.hour, dt.minute, dt.second))

    def convert_cv_qt(self, img):
        """Chuyển đổi từ ảnh OpenCV sang QPixmap."""
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(1050, 600, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)
    
    def info(self, text):
        in4 = self.let_bienso.text().strip()  # Loại bỏ khoảng trắng
        if not in4:  # Kiểm tra nếu in4 là chuỗi rỗng
            return  # Không làm gì nếu không có biển số xe
        
         # Kiểm tra nếu in4 là một chuỗi số hợp lệ
        if len(in4) < 2 or not in4[:2].isdigit():
            print(f"Biển số không hợp lệ: {in4}")  # In ra biển số không hợp lệ
            return  # Không làm gì nếu biển số không hợp lệ

        in5 = int(in4[0:2])
    
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
                self.let_tinh.show()  # Hiện thị tỉnh
                break  # Thoát vòng lặp sau khi tìm thấy

    def on_amthanh_clicked(self):  # Hàm phát âm thanh chỉ được gọi khi nhấn nút
        in4 = self.let_bienso.text()
        self.read_let_am_thanh(in4)

    def on_amthanh_clicked(self):
        """Chuyển đổi biển số xe thành giọng nói."""
        in4 = self.let_bienso.text()
        self.read_let_am_thanh(in4)

    def read_let_am_thanh(self, let_am_thanh):
        """Đọc to biển số xe sử dụng chuyển văn bản thành giọng nói."""
        chu_cai_va_so = {
            'A': 'A', 'B': 'Bê', 'C': 'Xê', 'D': 'Đê', 'E': 'E', 'F': 'Ép', 'G': 'Gờ',
            'H': 'Hát', 'J': 'Giê', 'K': 'Ca', 'L': 'Lờ', 'M': 'Mờ', 'N': 'Nờ',
            'P': 'Pê', 'R': 'Rờ', 'S': 'Ét', 'T': 'Tê', 'V': 'Vê', 'W': 'Vê Kép',
            'X': 'Ích', 'Y': 'I', 'Z': 'Dét', '0': 'Không', '1': 'Một', '2': 'Hai',
            '3': 'Ba', '4': 'Bốn', '5': 'Năm', '6': 'Sáu', '7': 'Bảy', '8': 'Tám', '9': 'Chín'
        }
        chuoi_phat_am = " ".join([chu_cai_va_so.get(ky_tu.upper(), ky_tu) for ky_tu in let_am_thanh])

        tts = gTTS(f"Biển số xe là {chuoi_phat_am}", lang='vi')
        tts.save("bien_so_xe.mp3")
        playsound("bien_so_xe.mp3")
        os.remove("bien_so_xe.mp3")

class CaptureVideo(QThread):
    signal = pyqtSignal(np.ndarray)
    updateBs = pyqtSignal(str)

    def __init__(self, video_path, yolo_detect, yolo_ocr,main_window):
        super(CaptureVideo, self).__init__()
        self.video_path = video_path
        self.yolo_detect = yolo_detect
        self.yolo_ocr = yolo_ocr
        self.main_window = main_window  # Lưu đối tượng MainWindow
        self.running = True
        self.cap = cv2.VideoCapture(self.video_path)
        self.paused = False  # Theo dõi trạng thái tạm dừng

    def run(self):
        """Xử lý từng khung hình của video để nhận diện biển số xe."""
        while self.running and self.cap.isOpened():
            if self.paused:
                continue  # Bỏ qua xử lý nếu đang tạm dừng
            
            ret, img = self.cap.read()
            if not ret:
                break

            plates = self.yolo_detect(img, size=640)
            list_plates = plates.pandas().xyxy[0].values.tolist()

            if list_plates:
                for plate in list_plates:
                    x = int(plate[0])
                    y = int(plate[1])
                    w = int(plate[2] - plate[0])
                    h = int(plate[3] - plate[1])
                    crop_img = img[y:y + h, x:x + w]
                    lp = helper.read_plate(self.yolo_ocr, crop_img)
                    if lp != "unknown" and lp.isalnum():  
                        self.updateBs.emit(lp)  # Gửi biển số xe nhận diện được
                        self.signal.emit(img)  # Gửi khung hình để hiển thị
                        self.main_window.info(lp)  # Gọi hàm info trong MainWindow với biển số
                        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(img, lp, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

            self.signal.emit(img)

    def stop(self):
        """Dừng xử lý video và giải phóng tài nguyên."""
        self.running = False
        self.cap.release()

    def pause(self):
        """Tạm dừng xử lý video."""
        self.paused = True

    def resume(self):
        """Tiếp tục xử lý video."""
        self.paused = False


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    widget = MainWindow()
    widget.show()
    sys.exit(app.exec_())
