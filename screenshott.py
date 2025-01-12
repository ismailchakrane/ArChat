from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget
from PyQt5.QtCore import QTimer
import sys
from PIL import Image
import pytesseract
from PyQt5.QtWidgets import QApplication
import tempfile
from multiprocessing import Process
import pyautogui
from fpdf import FPDF

class ScreenshotNotifier(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Notification")
        self.setGeometry(100, 100, 300, 100)

        layout = QVBoxLayout()
        self.label = QLabel("Capture d'écran commencera dans 3 secondes...", self)
        self.label.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(self.label)

        self.setLayout(layout)

    def start_countdown(self, duration=5, on_complete=None):
        self.remaining_time = duration
        self.on_complete = on_complete
        self.update_label()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_countdown)
        self.timer.start(1000)

    def update_countdown(self):
        self.remaining_time -= 1
        if self.remaining_time <= 0:
            self.timer.stop()
            self.close()
            if self.on_complete:
                self.on_complete()  # Appelle la fonction de capture d'écran
        else:
            self.update_label()

    def update_label(self):
        self.label.setText(f"Screenshot will start in {self.remaining_time} seconds...")

def save_text_as_temp_pdf(text):
    try:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf", dir="data")
        temp_path = temp_file.name
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, text.encode('latin-1', 'replace').decode('latin-1'))
        pdf.output(temp_path)
        return temp_path
    except Exception as e:
        print(f"Error creating temporary PDF file: {e}")
        return None


def start_screenshot():
    screenshot = pyautogui.screenshot()
    screenshot.save("screenshot.png")


def screenshot_notifier_process():
    app = QApplication(sys.argv)
    notifier = ScreenshotNotifier()
    notifier.show()
    notifier.start_countdown(on_complete=start_screenshot)
    sys.exit(app.exec_())


def show_screenshot_notifier():
    process = Process(target=screenshot_notifier_process)
    process.start()
    process.join()


def extract_text_from_image(image_path):
    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        print(f"Erreur lors de l'extraction de texte : {e}")
        return ""
