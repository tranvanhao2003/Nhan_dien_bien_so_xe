# Form implementation generated from reading ui file 'layout_video.ui'
#
# Created by: PyQt6 UI code generator 6.7.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Frame(object):
    def setupUi(self, Frame):
        Frame.setObjectName("Frame")
        Frame.resize(1538, 859)
        Frame.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.frame = QtWidgets.QFrame(parent=Frame)
        self.frame.setEnabled(True)
        self.frame.setGeometry(QtCore.QRect(30, 30, 1521, 841))
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.frame.setFont(font)
        self.frame.setMouseTracking(False)
        self.frame.setTabletTracking(False)
        self.frame.setAcceptDrops(False)
        self.frame.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.frame.setFrameShape(QtWidgets.QFrame.Shape.Box)
        self.frame.setFrameShadow(QtWidgets.QFrame.Shadow.Raised)
        self.frame.setMidLineWidth(0)
        self.frame.setObjectName("frame")
        self.label = QtWidgets.QLabel(parent=self.frame)
        self.label.setGeometry(QtCore.QRect(10, 10, 1491, 71))
        font = QtGui.QFont()
        font.setPointSize(22)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.ArrowCursor))
        self.label.setStyleSheet("background-color: rgb(255, 255, 255);\n"
"\n"
"color: rgb(170, 0, 0)")
        self.label.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label.setObjectName("label")
        self.frame_2 = QtWidgets.QFrame(parent=self.frame)
        self.frame_2.setGeometry(QtCore.QRect(-10, 80, 1511, 741))
        self.frame_2.setStyleSheet("background-color: rgb(85, 85, 0);\n"
"background-color: rgb(170, 170, 127);")
        self.frame_2.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Shadow.Raised)
        self.frame_2.setObjectName("frame_2")
        self.original_video = QtWidgets.QLabel(parent=self.frame_2)
        self.original_video.setGeometry(QtCore.QRect(20, 10, 1021, 711))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.original_video.setFont(font)
        self.original_video.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.original_video.setFrameShape(QtWidgets.QFrame.Shape.Box)
        self.original_video.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.original_video.setObjectName("original_video")
        self.btn_chonVideo = QtWidgets.QPushButton(parent=self.frame_2)
        self.btn_chonVideo.setGeometry(QtCore.QRect(1120, 120, 141, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.btn_chonVideo.setFont(font)
        self.btn_chonVideo.setStyleSheet("background-color: rgb(0, 170, 0);\n"
"color: rgb(255, 255, 255);")
        self.btn_chonVideo.setObjectName("btn_chonVideo")
        self.btn_nhandang = QtWidgets.QPushButton(parent=self.frame_2)
        self.btn_nhandang.setGeometry(QtCore.QRect(1300, 120, 141, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.btn_nhandang.setFont(font)
        self.btn_nhandang.setStyleSheet("background-color: rgb(255, 0, 0);\n"
"color: rgb(255, 255, 255);")
        self.btn_nhandang.setObjectName("btn_nhandang")
        self.label_4 = QtWidgets.QLabel(parent=self.frame_2)
        self.label_4.setGeometry(QtCore.QRect(1070, 240, 131, 51))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.label_4.setFont(font)
        self.label_4.setStyleSheet("background-color: rgb(238, 238, 238);")
        self.label_4.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.label_4.setObjectName("label_4")
        self.let_bienso = QtWidgets.QLineEdit(parent=self.frame_2)
        self.let_bienso.setGeometry(QtCore.QRect(1220, 240, 271, 51))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(False)
        font.setWeight(50)
        self.let_bienso.setFont(font)
        self.let_bienso.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.let_bienso.setText("")
        self.let_bienso.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.let_bienso.setObjectName("let_bienso")
        self.let_ngay = QtWidgets.QLineEdit(parent=self.frame_2)
        self.let_ngay.setGeometry(QtCore.QRect(1220, 440, 271, 51))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.let_ngay.setFont(font)
        self.let_ngay.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.let_ngay.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.let_ngay.setObjectName("let_ngay")
        self.let_gio = QtWidgets.QLineEdit(parent=self.frame_2)
        self.let_gio.setGeometry(QtCore.QRect(1220, 540, 271, 51))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.let_gio.setFont(font)
        self.let_gio.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.let_gio.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.let_gio.setObjectName("let_gio")
        self.label_9 = QtWidgets.QLabel(parent=self.frame_2)
        self.label_9.setGeometry(QtCore.QRect(1070, 340, 131, 51))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.label_9.setFont(font)
        self.label_9.setStyleSheet("background-color: rgb(238, 238, 238);")
        self.label_9.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.label_9.setObjectName("label_9")
        self.label_6 = QtWidgets.QLabel(parent=self.frame_2)
        self.label_6.setGeometry(QtCore.QRect(1070, 440, 131, 51))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.label_6.setFont(font)
        self.label_6.setStyleSheet("background-color: rgb(238, 238, 238);")
        self.label_6.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(parent=self.frame_2)
        self.label_7.setGeometry(QtCore.QRect(1070, 540, 131, 51))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.label_7.setFont(font)
        self.label_7.setStyleSheet("background-color: rgb(238, 238, 238);")
        self.label_7.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.label_7.setObjectName("label_7")
        self.let_tinh = QtWidgets.QLineEdit(parent=self.frame_2)
        self.let_tinh.setGeometry(QtCore.QRect(1220, 340, 271, 51))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(False)
        font.setWeight(50)
        self.let_tinh.setFont(font)
        self.let_tinh.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.let_tinh.setText("")
        self.let_tinh.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.let_tinh.setObjectName("let_tinh")
        self.btn_amthanh = QtWidgets.QPushButton(parent=self.frame_2)
        self.btn_amthanh.setGeometry(QtCore.QRect(1200, 640, 151, 51))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.btn_amthanh.setFont(font)
        self.btn_amthanh.setStyleSheet("background-color: rgb(0, 85, 255);\n"
"color: rgb(255, 255, 255);")
        self.btn_amthanh.setAutoRepeat(False)
        self.btn_amthanh.setObjectName("btn_amthanh")
        self.btn_chonVideo_2 = QtWidgets.QPushButton(parent=self.frame_2)
        self.btn_chonVideo_2.setGeometry(QtCore.QRect(1120, 50, 321, 51))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.btn_chonVideo_2.setFont(font)
        self.btn_chonVideo_2.setStyleSheet("color: rgb(255, 255, 255);\n"
"background-color: rgb(255, 170, 127);")
        self.btn_chonVideo_2.setObjectName("btn_chonVideo_2")

        self.retranslateUi(Frame)
        QtCore.QMetaObject.connectSlotsByName(Frame)

    def retranslateUi(self, Frame):
        _translate = QtCore.QCoreApplication.translate
        Frame.setWindowTitle(_translate("Frame", "Frame"))
        self.label.setText(_translate("Frame", "NHẬN DIỆN BIỂN SỐ XE"))
        self.original_video.setText(_translate("Frame", "Video gốc"))
        self.btn_chonVideo.setText(_translate("Frame", "Bắt đầu"))
        self.btn_nhandang.setText(_translate("Frame", "Dừng"))
        self.label_4.setText(_translate("Frame", "Biển số:"))
        self.label_9.setText(_translate("Frame", "Tỉnh:"))
        self.label_6.setText(_translate("Frame", "Ngày:"))
        self.label_7.setText(_translate("Frame", "Giờ:"))
        self.btn_amthanh.setText(_translate("Frame", "Âm thanh"))
        self.btn_chonVideo_2.setText(_translate("Frame", "Chọn video"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Frame = QtWidgets.QFrame()
    ui = Ui_Frame()
    ui.setupUi(Frame)
    Frame.show()
    sys.exit(app.exec())
