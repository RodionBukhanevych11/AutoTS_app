import sys
from PyQt5.QtWidgets import QApplication
from windows import MainWindow
import argparse
import sys
import os
import warnings
warnings.simplefilter('ignore')

#filename='data/dataset.csv'

def create_arg_parser():
    parser = argparse.ArgumentParser(description='####')
    parser.add_argument('inputFile',
                    help='Path to the input file with time series')
    return parser


if __name__ == '__main__':
    try:
        arg_parser = create_arg_parser()
        parsed_args = arg_parser.parse_args(sys.argv[1:])
        if os.path.exists(parsed_args.inputFile):
            filename = parsed_args.inputFile
        else:
            print("Файл несуществует")
        app = QApplication(sys.argv)
        window = MainWindow(filename)
        window.show()
        sys.exit(app.exec_())
    except:
        print("Правильный ввод: <py main.py dataset.csv>")