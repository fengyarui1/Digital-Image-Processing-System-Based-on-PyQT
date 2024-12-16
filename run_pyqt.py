import sys
import cv2 as cv
import qdarkstyle
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import QFileDialog, QApplication, QMainWindow
from view import Ui_MainWindow
from PIL import Image

class PyQtMainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        # 连接滑块的valueChanged信号到自定义的槽函数self.on_horizontalSlider_valueChanged
        self.horizontalSlider.valueChanged.connect(self.on_horizontalSlider_valueChanged)
        self.camera = cv.VideoCapture(0)
        self.is_camera_opened = False  # 摄像头有没有打开标记
        self._timer = QtCore.QTimer(self) # 定时器：30ms捕获一帧
        self._timer.timeout.connect(self._queryFrame)
        self._timer.setInterval(30)
        self.btnCamera.clicked.connect(self.btnOpenCamera_Clicked)
        self.btnCapture.clicked.connect(self.btnCapture_Clicked)
        self.btnOpen.clicked.connect(self.btnReadImage_Clicked)
        self.btnGray.clicked.connect(self.btnGray_Clicked)
        self.btnBin.clicked.connect(self.btnThreshold_Clicked)
        self.btnCanny.clicked.connect(self.btnCanny_Clicked)
        self.btnRotate.clicked.connect(self.btnRotate_Clicked)
        self.btnGaussianBlur.clicked.connect(self.btnGaussianBlur_Clicked)
        self.btnEnhance.clicked.connect(self.btnEnhance_Clicked)
        self.btnCompress.clicked.connect(self.btnCompress_Clicked)
        self.btnFaceDetect.clicked.connect(self.btnFaceDetect_Clicked)
    @pyqtSlot(int)
    def on_horizontalSlider_valueChanged(self, value):
        self.slider_value=value
        #print("滑块当前的值为:", value)

    def btnOpenCamera_Clicked(self):
        '''
        打开和关闭摄像头
        '''
        self.is_camera_opened = ~self.is_camera_opened
        if self.is_camera_opened:
            self.btnCamera.setText("关闭摄像头")
            self._timer.start()
        else:
            self.btnCamera.setText("打开摄像头")
            self._timer.stop()

    def btnCapture_Clicked(self):
        '''
        捕获图片
        '''
        # 摄像头未打开，不执行任何操作
        if not self.is_camera_opened:
            return

        self.captured = self.frame

        # 后面这几行代码几乎都一样，可以尝试封装成一个函数
        rows, cols, channels = self.captured.shape
        bytesPerLine = channels * cols
        # Qt显示图片时，需要先转换成QImgage类型
        QImg = QImage(self.captured.data, cols, rows, bytesPerLine, QImage.Format_RGB888)
        self.labelCapture.setPixmap(QPixmap.fromImage(QImg).scaled(
            self.labelCapture.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def btnReadImage_Clicked(self):
        '''
        从本地读取图片
        '''
        # 打开文件选取对话框
        filename,  _ = QFileDialog.getOpenFileName(self, '打开图片')
        if filename:
            self.captured = cv.imread(rf"{filename}")
            # 图像路径不能包括中文
            # OpenCV图像以BGR通道存储，显示时需要从BGR转到RGB
            self.captured = cv.cvtColor(self.captured, cv.COLOR_BGR2RGB)

            rows, cols, channels = self.captured.shape
            bytesPerLine = channels * cols # 该参数指定了图像每行像素所占用的字节数
            QImg = QImage(self.captured.data, cols, rows, bytesPerLine, QImage.Format_RGB888)
            self.labelCapture.setPixmap(QPixmap.fromImage(QImg).scaled(
                self.labelCapture.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def btnGray_Clicked(self):
        '''
        灰度化
        '''
        # 如果没有捕获图片，则不执行操作
        if not hasattr(self, "captured"):
            return

        self.captured_gray = cv.cvtColor(self.captured, cv.COLOR_RGB2GRAY)

        rows, columns = self.captured_gray.shape
        bytesPerLine = columns
        # 灰度图是单通道，所以需要用Format_Indexed8
        QImg = QImage(self.captured_gray.data, columns, rows, bytesPerLine, QImage.Format_Indexed8)
        self.labelResult.setPixmap(QPixmap.fromImage(QImg).scaled(
            self.labelResult.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def btnThreshold_Clicked(self):
        '''
        阈值分割
        '''
        if not hasattr(self, "captured"):
            return
        
        if self.captured.ndim == 3:
            source = cv.cvtColor(self.captured, cv.COLOR_RGB2GRAY)
        else:
            source = self.captured
        _, self.captured_bin = cv.threshold(source, 127, 255, cv.THRESH_BINARY)

        rows, columns = self.captured_bin.shape
        bytesPerLine = columns
        # 阈值分割图也是单通道，也需要用Format_Indexed8
        QImg = QImage(self.captured_bin.data, columns, rows, bytesPerLine, QImage.Format_Indexed8)
        self.labelResult.setPixmap(QPixmap.fromImage(QImg).scaled(
            self.labelResult.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def btnCanny_Clicked(self):
        '''
        canny边缘检测
        '''
        if not hasattr(self, "captured"):
            return
        
        if self.captured.ndim == 3:
            source = cv.cvtColor(self.captured, cv.COLOR_RGB2GRAY)
        else:
            source = self.captured

        self.captured_bin = cv.Canny(source, threshold1=20, threshold2=200)
        rows, columns = self.captured_bin.shape
        bytesPerLine = columns
        QImg = QImage(self.captured_bin.data, columns, rows, bytesPerLine, QImage.Format_Indexed8)
        self.labelResult.setPixmap(QPixmap.fromImage(QImg).scaled(
            self.labelResult.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def btnRotate_Clicked(self):
        '''
        旋转图像
        '''
        if not hasattr(self, "captured"):
            return

            # 设定旋转的角度（以度为单位）
        angle = self.slider_value  # 你可以根据需要改变这个值
        print("使用滑块的值作为旋转角度:", angle)
        # 获取图像的中心点
        (h, w) = self.captured.shape[:2]
        center = (w // 2, h // 2)

        # 获取旋转矩阵
        M = cv.getRotationMatrix2D(center, angle, 1.0)

        # 执行仿射变换（旋转）
        self.captured_rotated = cv.warpAffine(self.captured, M, (w, h))

        # 将OpenCV图像转换为QImage
        if len(self.captured_rotated.shape) == 2:  # 灰度图像
            bytes_per_line = self.captured_rotated.shape[1]
            QImg = QImage(self.captured_rotated.data, self.captured_rotated.shape[1], self.captured_rotated.shape[0],
                          bytes_per_line, QImage.Format_Indexed8)
            QImg.setColorTable([QColor(i, i, i).rgb() for i in range(256)])  # 设置灰度图像的色表
        else:  # 彩色图像
            bytes_per_line = 3 * self.captured_rotated.shape[1]
            QImg = QImage(self.captured_rotated.data, self.captured_rotated.shape[1], self.captured_rotated.shape[0],
                          bytes_per_line, QImage.Format_RGB888)
        self.labelResult.setPixmap(QPixmap.fromImage(QImg).scaled(
            self.labelResult.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def btnGaussianBlur_Clicked(self):
        """
        对捕获的图像进行高斯去噪并显示
        """
        if not hasattr(self, "captured"):
            return

        # 使用高斯滤波进行去噪，这里使用5x5的核，标准差设为0（自动根据核大小计算），可根据实际调整参数
        denoised_image = cv.GaussianBlur(self.captured, (5, 5), 0)

        rows, columns, channels = denoised_image.shape
        bytesPerLine = columns * channels
        # 根据图像通道情况选择合适的QImage格式，这里考虑彩色和灰度两种常见情况
        if channels == 1:
            # 灰度图情况，使用Format_Indexed8
            QImg = QImage(denoised_image.data, columns, rows, bytesPerLine, QImage.Format_Indexed8)
        else:
            # 彩色图情况，使用Format_RGB888
            QImg = QImage(denoised_image.data, columns, rows, bytesPerLine, QImage.Format_RGB888)

        self.labelResult.setPixmap(QPixmap.fromImage(QImg).scaled(
            self.labelResult.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def btnEnhance_Clicked(self):
        """
        对捕获的图像进行图像增强（这里以直方图均衡化为例）并显示
        """
        # 如果没有捕获图片，则不执行操作
        if not hasattr(self, "captured"):
            return

        # 判断图像是彩色还是灰度，分别进行处理
        if len(self.captured.shape) == 2:
            # 灰度图像，直接进行直方图均衡化
            enhanced_image = cv.equalizeHist(self.captured)
        else:
            # 彩色图像，先转换到YUV颜色空间，对亮度通道Y进行直方图均衡化，再转换回RGB
            yuv_image = cv.cvtColor(self.captured, cv.COLOR_RGB2YUV)
            yuv_image[:, :, 0] = cv.equalizeHist(yuv_image[:, :, 0])
            enhanced_image = cv.cvtColor(yuv_image, cv.COLOR_YUV2RGB)

        rows, columns, channels = enhanced_image.shape
        bytesPerLine = columns * channels
        # 根据图像通道情况选择合适的QImage格式，这里考虑彩色和灰度两种常见情况
        if channels == 1:
            # 灰度图情况，使用Format_Indexed8
            QImg = QImage(enhanced_image.data, columns, rows, bytesPerLine, QImage.Format_Indexed8)
        else:
            # 彩色图情况，使用Format_RGB888
            QImg = QImage(enhanced_image.data, columns, rows, bytesPerLine, QImage.Format_RGB888)

        self.labelResult.setPixmap(QPixmap.fromImage(QImg).scaled(
            self.labelResult.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def btnCompress_Clicked(self):
        """
        对捕获的图像进行图像压缩并显示压缩后的图像（这里使用Pillow库实现简单的JPEG格式压缩示例）
        """
        # 如果没有捕获图片，则不执行操作
        if not hasattr(self, "captured"):
            return

        # 将OpenCV格式的图像（numpy数组形式）转换为PIL Image对象，便于使用PIL进行压缩操作
        pil_image = Image.fromarray(cv.cvtColor(self.captured, cv.COLOR_BGR2RGB))

        # 设置压缩质量，这里示例设为50，可以根据实际需求调整（范围0 - 100）
        quality = 50
        output_image_path = "compressed_image.jpg"
        try:
            # 保存图像，以指定的质量参数进行JPEG格式压缩
            pil_image.save(output_image_path, format='JPEG', quality=quality)
            # 再读取压缩后的图像数据，转换为可以在Qt界面显示的格式
            compressed_image = cv.imread(output_image_path)
            rows, columns, channels = compressed_image.shape
            bytesPerLine = columns * channels
            if channels == 1:
                # 灰度图情况，使用Format_Indexed8
                QImg = QImage(compressed_image.data, columns, rows, bytesPerLine, QImage.Format_Indexed8)
            else:
                # 彩色图情况，使用Format_RGB888
                QImg = QImage(compressed_image.data, columns, rows, bytesPerLine, QImage.Format_RGB888)

            self.labelResult.setPixmap(QPixmap.fromImage(QImg).scaled(
                self.labelResult.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        except Exception as e:
            print(f"图像压缩过程出现错误: {e}")

    def btnFaceDetect_Clicked(self):
        """
        对捕获的图像进行人脸检测，并在图像上标记出人脸区域后显示
        """
        # 如果没有捕获图片，则不执行操作
        if not hasattr(self, "captured"):
            return

        # 加载人脸检测的级联分类器（这里使用默认的 frontalface 分类器，需确保文件存在）
        face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

        # 将图像转换为灰度图，因为人脸检测通常在灰度图上进行效果更好且速度更快
        gray_image = cv.cvtColor(self.captured, cv.COLOR_BGR2GRAY)

        # 进行人脸检测，返回检测到的人脸区域坐标列表（格式为 [x, y, w, h]，x,y为左上角坐标，w为宽度，h为高度）
        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # 在原始彩色图像上绘制矩形框标记出人脸区域
        for (x, y, w, h) in faces:
            cv.rectangle(self.captured, (x, y), (x + w, y + h), (0, 255, 0), 2)

        rows, columns, channels = self.captured.shape
        bytesPerLine = columns * channels
        if channels == 1:
            # 灰度图情况，使用Format_Indexed8
            QImg = QImage(self.captured.data, columns, rows, bytesPerLine, QImage.Format_Indexed8)
        else:
            # 彩色图情况，使用Format_RGB888
            QImg = QImage(self.captured.data, columns, rows, bytesPerLine, QImage.Format_RGB888)

        self.labelResult.setPixmap(QPixmap.fromImage(QImg).scaled(
            self.labelResult.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    @QtCore.pyqtSlot()
    def _queryFrame(self):
        '''
        循环捕获图片
        '''
        ret, self.frame = self.camera.read()

        img_rows, img_cols, channels = self.frame.shape
        bytesPerLine = channels * img_cols

        cv.cvtColor(self.frame, cv.COLOR_BGR2RGB, self.frame)
        QImg = QImage(self.frame.data, img_cols, img_rows, bytesPerLine, QImage.Format_RGB888)
        self.labelCamera.setPixmap(QPixmap.fromImage(QImg).scaled(
            self.labelCamera.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))


def convert(ui):
    inp = ui.lineEdit.text()
    result = float(inp) * 6.71
    ui.lineEdit_2.setText(str(result))



if __name__ == '__main__':
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling) # 用于解决按钮文字显示不全，完成屏幕自适应
    app = QApplication(sys.argv)
    #app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5()) # 界面美化
    win = PyQtMainWindow()
    win.show()
    sys.exit(app.exec_())
