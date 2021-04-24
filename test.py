import sys
import time
 
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QColor
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import PyQt5.QtGui as QtGui 
from PyQt5.QtCore import QObject, QThread, pyqtSignal


def print_func():
    print("A")
    time.sleep(5)

def print_func2():
    print("B")
    time.sleep(5) 
 
    
    def clicked(self):
        item = self.listwidget.currentItem()
        self.widget_item = item.text()

    def plot_graphs(self):
        self.axes[0][0].clear()
        self.axes[1][0].clear()
        self.axes[1][1].clear()
        self.axes[0][1].clear()
        self.canvas.draw()
 
# Объект, который будет перенесён в другой поток для выполнения кода
class A(QtCore.QObject):
    finished = pyqtSignal()
    result = pyqtSignal(object)
    def __init__(self):
        super().__init__()
        self.a = 1
    # метод, который будет выполнять алгоритм в другом потоке
    def run(self):
        print_func()
        self.finished.emit()
        self.result.emit(1)

 
class B(QtCore.QObject):
    finished = pyqtSignal()
    result = pyqtSignal(object)
    # метод, который будет выполнять алгоритм в другом потоке
    def run(self):
        print_func2()
        self.finished.emit() 
        self.result.emit(2)

class MyWindow(QtWidgets.QWidget):
 
    def __init__(self, parent=None):
        super().__init__()
        self.view = QTableView()
        self.setWindowTitle("TimeSeriesPredictionSystem")
        self.Button1 = QPushButton("A")
        self.Button2 = QPushButton("B")
        self.Button1.setFixedSize(300, 80)
        self.Button2.setFixedSize(300, 80)
        self.layout = QGridLayout()
        self.layout.addWidget(self.Button1, 0, 0)
        self.layout.addWidget(self.Button2, 1, 0)
        self.layout.setHorizontalSpacing(5)
        self.layout.setVerticalSpacing(5)
        self.Button1.clicked.connect(self.start_printA)
        self.Button2.clicked.connect(self.start_printB)
        self.Button1.setFont(QtGui.QFont("Serif", 10, QtGui.QFont.Bold))
        self.Button2.setFont(QtGui.QFont("Serif", 10, QtGui.QFont.Bold))
        self.setLayout(self.layout)
        # запустим поток
    def print_output(self, s):
        print(s)

    def start_printA(self):
        self.thread = QtCore.QThread()
        self.A = A()
        self.A.moveToThread(self.thread)
        self.thread.started.connect(self.A.run)
        self.A.finished.connect(self.thread.quit)
        self.A.finished.connect(self.A.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()
        self.A.result.connect(self.print_output)
        self.Button1.setEnabled(False)
        self.Button2.setEnabled(False)
        self.thread.finished.connect(
            lambda: self.enableButton()
        )

    def enableButton(self,mode = True):
        self.Button1.setEnabled(mode)
        self.Button2.setEnabled(mode)
    
    def start_printB(self):
        self.thread2 = QtCore.QThread()
        self.B = B()
        self.B.moveToThread(self.thread2)
        self.thread2.started.connect(self.B.run)
        self.B.finished.connect(self.thread2.quit)
        self.B.finished.connect(self.B.deleteLater)
        self.thread2.finished.connect(self.thread2.deleteLater)
        self.thread2.start()
        self.Button2.setEnabled(False)
        self.thread2.finished.connect(
            lambda: self.Button2.setEnabled(True)
        )

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec())
 