import sys
import pandas as pd
import numpy as np
from functools import reduce
from PyQt5.QtWidgets import QApplication
import time
from windows import MainWindow
import warnings
warnings.simplefilter('ignore')

filename='data/dataset.csv'

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow(filename)
    window.show()
    sys.exit(app.exec_())