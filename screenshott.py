from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget
from PyQt5.QtCore import QTimer
import sys

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
        """
        Démarre un compte à rebours et exécute une fonction une fois terminé.
        :param duration: Durée en secondes.
        :param on_complete: Fonction à exécuter après le compte à rebours.
        """
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
        self.label.setText(f"Capture d'écran commencera dans {self.remaining_time} secondes...")