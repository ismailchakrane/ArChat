from PyQt5.QtCore import Qt, QRect
from PyQt5.QtGui import QPainter, QPen
from PyQt5.QtWidgets import QApplication, QMainWindow
import pyautogui
import sys
import os


class AdjustableScreenshotWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Adjust Screenshot Area")
        self.setGeometry(0, 0, QApplication.primaryScreen().size().width(),
                         QApplication.primaryScreen().size().height())
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setWindowOpacity(0.3)
        self.begin = None
        self.end = None
        self.setCursor(Qt.CrossCursor)
        self.showFullScreen()

    def paintEvent(self, event):
        if self.begin and self.end:
            rect = QRect(self.begin, self.end)
            painter = QPainter(self)
            painter.setPen(QPen(Qt.red, 2, Qt.SolidLine))
            painter.drawRect(rect)

    def mousePressEvent(self, event):
        self.begin = event.pos()
        self.end = self.begin
        self.update()

    def mouseMoveEvent(self, event):
        self.end = event.pos()
        self.update()

    def mouseReleaseEvent(self, event):
        x1 = min(self.begin.x(), self.end.x())
        y1 = min(self.begin.y(), self.end.y())
        x2 = max(self.begin.x(), self.end.x())
        y2 = max(self.begin.y(), self.end.y())

        self.capture_screenshot(x1, y1, x2, y2)
        self.close()

    def capture_screenshot(self, x1, y1, x2, y2):
        screenshot = pyautogui.screenshot()
        cropped = screenshot.crop((x1, y1, x2, y2))
        cropped.save("adjusted_screenshot.png")
        print("Screenshot saved as 'adjusted_screenshot.png'")


def take_adjustable_screenshot():
    app = QApplication(sys.argv)
    window = AdjustableScreenshotWindow()
    app.exec_()
    return "adjusted_screenshot.png"
