from PyQt5 import QtCore, QtGui, QtWidgets
import sys
from PyQt5.QtCore import QSize, pyqtSlot
from PyQt5.QtGui import QImage, QPalette, QBrush, QPixmap, QMovie
import Navigator

from PyQt5.QtWidgets import *
class StartButton(QtWidgets.QPushButton):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setMinimumSize(60, 60)

        self.color1 = QtGui.QColor(240, 53, 218)
        self.color2 = QtGui.QColor(61, 217, 245)

        self._animation = QtCore.QVariantAnimation(
            self,
            valueChanged=self._animate,
            startValue=0.00001,
            endValue=0.9999,
            duration=400
        )

    def _animate(self, value):
        qss = """
            font: 75 10pt "Microsoft YaHei UI";
            font-weight: bold;
            color: rgb(255, 255, 255);
            border-style: solid;
            border-radius:21px;
        """
        grad = "background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, stop:0 {color1}, stop:{value} {color2}, stop: 1.0 {color1});".format(
            color1=self.color1.name(), color2=self.color2.name(), value=value
        )
        qss += grad
        self.setStyleSheet(qss)

    def enterEvent(self, event):
        self._animation.setDirection(QtCore.QAbstractAnimation.Forward)
        self._animation.start()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._animation.setDirection(QtCore.QAbstractAnimation.Backward)
        self._animation.start()
        super().enterEvent(event)

@pyqtSlot()
def on_click(self):
    Navigator.main()

class MainWindow(QWidget):
    
    def __init__(self):
        QWidget.__init__(self)
        self.resize(800, 900)
        oImage = QImage("../gui/test.jpg")
        sImage = oImage.scaled(QSize(800, 900))      
        palette = QPalette()
        palette.setBrush(10, QBrush(sImage))
        lay = QtWidgets.QVBoxLayout(self)
        h_lay = QtWidgets.QHBoxLayout()
        h_lay.addStretch()
        h_lay1 = QtWidgets.QHBoxLayout()
        h_lay1.addStretch()
        button = StartButton()
        button.setText("Start")
        button.setDefault(True)
        button.setCheckable(True)
        button.clicked.connect(on_click)
        lay.addStretch(1)
        lay.setSpacing(50)
        self.setPalette(palette)
        seye = QLabel() 
        eye = QMovie('../gui/eyey1.gif') 
        seye.setMovie(eye)
        eye.start()
        slog = QLabel() 
        sign = QPixmap('../gui/text.png') 
        slog.setPixmap(sign)
        h_lay.addWidget(slog)
        h_lay.addStretch()
        h_lay1.addWidget(seye)
        h_lay1.addStretch()
        lay.addLayout(h_lay)
        lay.addLayout(h_lay1)
        lay.addWidget(button)
        self.show()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    oMainwindow = MainWindow()
    sys.exit(app.exec_())
