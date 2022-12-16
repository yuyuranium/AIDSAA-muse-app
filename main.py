from PyQt5.QtWidgets import *
from PyQt5 import QtGui
from UI import Ui_MuseApp

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import numpy as np
from audio_recorder import AudioRecorder
from muse_classifier import MuseClassifier

import sys
import time
import threading

class PltWidget(QWidget):
    def __init__(self, parent=None):
        super(PltWidget, self).__init__(parent)
        self.figure = Figure()
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.axis = self.figure.add_subplot(111)
        QVBoxLayout(self).addWidget(self.canvas)

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        # UI define
        self.ui = Ui_MuseApp()
        self.ui.setupUi(self)
        self.ui.label.setText('Press start')
        self.ui.pushButton.clicked.connect(self.on_click)
        self.plt_widget = PltWidget()
        QVBoxLayout(self.ui.widget).addWidget(self.plt_widget)
        self.plt_widget.axis.axis('off')
        self.setFixedHeight(262)
        self.setFixedWidth(322)
        self.setWindowIcon(QtGui.QIcon('logo.png'))
        self.setWindowTitle('MuseApp')

        # Control
        self.data_valid = False
        self.model_ready = False
        self.started = False
        self.recording_stopped = False
        self.predicting_stopped = False

        # Data
        self.a = AudioRecorder()
        self.m = MuseClassifier('model_loss.h5')
        self.npdata = None
        self.d = {0: 'Joy', 1: 'Tension', 2: 'Sadness', 3: 'Peacefulness'}

    def on_click(self):
        if not self.started:
            self.ui.label.setText('Loading CUDA')
            self.started = True
            recording_thread = threading.Thread(target=self.start_recording,
                                                daemon=True)
            predicting_thread = threading.Thread(target=self.start_predicting,
                                                 daemon=True)
            self.recording_stopped = False
            self.predicting_stopped = False
            recording_thread.start()
            predicting_thread.start()
        else:
            self.ui.label.setText('Stopped')
            self.ui.pushButton.setEnabled(False)
            self.started = False
            time.sleep(1)

            # wait for both thread stopped
            while not self.recording_stopped or not self.predicting_stopped:
                pass
            self.ui.pushButton.setEnabled(True)
            self.ui.label.setText('Press start')
        self.ui.pushButton.setText('stop' if self.started else 'start')

    def start_recording(self):
        seconds = 2
        while self.started:
            self.a.start(seconds=seconds)
            self.npdata = self.a.get_npdata()
            self.data_valid = True

            # wait for model ready
            while not self.model_ready:
                if not self.started:
                    break

            # plot data
            self.plt_widget.axis.clear()
            times = np.linspace(0, seconds, self.npdata.shape[0])
            self.plt_widget.axis.axis('off')
            self.plt_widget.axis.set_xlim(0, seconds)
            self.plt_widget.axis.plot(times, self.npdata)
            self.plt_widget.canvas.draw()

            self.data_valid = False

        self.a.clear()
        print('Recording thread stopped')
        self.recording_stopped = True

    def start_predicting(self):
        while self.started:
            self.model_ready = True

            # wait for data valid
            while not self.data_valid:
                if not self.started:
                    break

            # get model input
            raw_data = np.array(self.npdata, dtype=float)
            self.model_ready = False
            p = self.m.predict(raw_data)

            # Show result to App
            self.ui.label.setText(self.d[p])

        print('Predicting thread stopped')
        self.predicting_stopped = True


if __name__ == '__main__':
    app = QApplication([])
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
