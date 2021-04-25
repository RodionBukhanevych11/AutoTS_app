from PyQt5 import QtWidgets,QtWebEngineWidgets,QtCore
from PyQt5.QtWidgets import *
import PyQt5.QtGui as QtGui
from PyQt5.QtCore import QAbstractTableModel, Qt
from data import *
import pandas as pd
import numpy as np
import time
from validation import train_valid_split,metrics
from prediction_methods import varma_prediction,arma_prediction,boosting_validation,boosting_prediction
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from qasync import asyncSlot
from PyQt5.QtCore import QObject, QThread, pyqtSignal
import warnings
warnings.simplefilter('ignore')

class MainWindow(QWidget):
    def __init__(self,file_path):
        super().__init__()
        self.df= read_df(file_path)
        self.stationary_df, self.inverting_times = make_stationary(self.df)
        self.w_prediction = Window_predictions(self.df,self.stationary_df,self.inverting_times)
        self.w_graphs = Window_graphs(self.df)
        self.model_df = pandasModel(self.df)
        self.view = QTableView()
        self.setWindowTitle("TimeSeriesPredictionSystem")
        self.Button1 = QPushButton("Датасет")
        self.Button2 = QPushButton("Графики")
        self.Button3 = QPushButton("Предсказание")
        self.Button1.setFixedSize(300, 80)
        self.Button2.setFixedSize(300, 80)
        self.Button3.setFixedSize(300, 80)
        self.layout = QGridLayout()
        self.layout.addWidget(self.Button1, 0, 0)
        self.layout.addWidget(self.Button2, 0, 1)
        self.layout.addWidget(self.Button3, 0, 2)
        self.layout.setHorizontalSpacing(5)
        self.layout.setVerticalSpacing(5)
        self.Button1.clicked.connect(self.dataset_func)
        self.Button2.clicked.connect(self.show_graphs)
        self.Button3.clicked.connect(self.show_predict)
        self.Button1.setFont(QtGui.QFont("Serif", 10, QtGui.QFont.Bold))
        self.Button2.setFont(QtGui.QFont("Serif", 10, QtGui.QFont.Bold))
        self.Button3.setFont(QtGui.QFont("Serif", 10, QtGui.QFont.Bold))
        self.setLayout(self.layout)

    def show_graphs(self):
        self.w_graphs.show()

    def show_predict(self):
        self.w_prediction.show()

    def dataset_func(self):
        self.show_tables(self.view,self.df)

    def plot_func(self):
        plot_timeseries(self.df)

    def show_tables(self,view,df):
        df = pandasModel(df)
        view.setModel(df)
        view.move(600, 500)
        view.resize(800,600)
        view.resizeColumnsToContents()
        view.show()

class Window_graphs(QWidget):
    def __init__(self,df):
        super().__init__()
        self.df = df
        self.setWindowTitle("Графики")
        self.Button1 = QPushButton("Plot")
        self.label = QLabel('Выберите временной ряд', self)
        self.Button1.setFixedSize(200, 50)
        self.label.setFixedSize(300,50)
        self.fig, self.axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 16))
        self.canvas = FigureCanvas(self.fig)
        self.listwidget = QListWidget()
        self.listwidget.setFixedSize(300,100)
        for i,col in enumerate(self.df.columns):
            self.listwidget.insertItem(i,col)
        layout = QGridLayout()
        layout.addWidget(self.Button1, 2, 0)
        layout.addWidget(self.label, 0, 0)
        layout.addWidget(self.listwidget, 1, 0)
        layout.addWidget(self.canvas, 3, 0)
        layout.setHorizontalSpacing(10)
        layout.setVerticalSpacing(10)
        self.Button1.clicked.connect(self.plot_graphs)
        self.listwidget.clicked.connect(self.clicked)
        self.Button1.setFont(QtGui.QFont("Serif", 10, QtGui.QFont.Bold))
        self.label.setFont(QtGui.QFont("Serif", 10, QtGui.QFont.Bold))
        self.listwidget.setFont(QtGui.QFont("Serif", 10, QtGui.QFont.Bold))
        self.widget_item = None
        self.setLayout(layout)
    
    def clicked(self):
        item = self.listwidget.currentItem()
        self.widget_item = item.text()

    def plot_graphs(self):
        self.axes[0][0].clear()
        self.axes[1][0].clear()
        self.axes[1][1].clear()
        self.axes[0][1].clear()
        plot_correlogram(self.df[self.widget_item],title = self.widget_item,fig = self.fig, axes = self.axes)
        self.canvas.draw()


class Prediction_graphs(QWidget):
    def __init__(self,df,val_df,predict_df):
        super().__init__()
        self.df = df
        self.stationary_df, self.inverting_times = make_stationary(self.df)
        self.val_df = val_df
        self.predict_df = predict_df
        self.methods = ["VARMA","BoDT","LSTM"]
        self.setWindowTitle("Графики")
        self.Button1 = QPushButton("Plot")
        self.label = QLabel('Выберите временной ряд', self)
        self.method = QLabel('Выберите метод', self)
        self.Button1.setFixedSize(200, 50)
        self.label.setFixedSize(300,50)
        self.fig, (self.ax1,self.ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 16))
        self.canvas = FigureCanvas(self.fig)
        self.listwidget = QListWidget()
        self.listwidget.setFixedSize(300,100)
        self.methodwidget = QListWidget()
        self.methodwidget.setFixedSize(300,100)
        for i,col in enumerate(self.df.columns):
            self.listwidget.insertItem(i,col)
        for i,method in enumerate(self.methods):
            self.methodwidget.insertItem(i,method)
        layout = QGridLayout()
        layout.addWidget(self.label, 0, 0)
        layout.addWidget(self.method, 0, 1)
        layout.addWidget(self.listwidget, 1, 0)
        layout.addWidget(self.methodwidget, 1, 1)
        layout.addWidget(self.Button1, 2, 0)
        layout.addWidget(self.canvas, 3, 0)
        layout.setHorizontalSpacing(10)
        layout.setVerticalSpacing(10)
        self.Button1.clicked.connect(self.plot_graphs)
        self.listwidget.clicked.connect(self.clicked)
        self.methodwidget.clicked.connect(self.clicked_method)
        self.Button1.setFont(QtGui.QFont("Serif", 9, QtGui.QFont.Bold))
        self.label.setFont(QtGui.QFont("Serif", 9, QtGui.QFont.Bold))
        self.method.setFont(QtGui.QFont("Serif", 9, QtGui.QFont.Bold))
        self.listwidget.setFont(QtGui.QFont("Serif", 9, QtGui.QFont.Bold))
        self.methodwidget.setFont(QtGui.QFont("Serif", 9, QtGui.QFont.Bold))
        self.widget_item = None
        self.methodwidget_item=None
        self.setLayout(layout)
    
    def clicked(self):
        item = self.listwidget.currentItem()
        self.widget_item = item.text()

    def clicked_method(self):
        item = self.methodwidget.currentItem()
        self.methodwidget_item = item.text()

    def plot_graphs(self):
            if self.predict_df[self.methodwidget_item] is not None and self.val_df[self.methodwidget_item] is not None:
                self.ax1.clear()
                self.ax2.clear()
                plot_pred_graphs(self.df[self.widget_item],self.predict_df[self.methodwidget_item][self.widget_item],self.val_df[self.methodwidget_item][self.widget_item],self.methodwidget_item,title = self.widget_item,fig = self.fig, ax1 = self.ax1,ax2=self.ax2)
                self.canvas.draw()
            else:
                self.warning_box("Сначала сделайте предсказания!")
        #except Exception as e:
            #self.warning_box("Выберите метод и ряд!")
            #print(e)

    def warning_box(self,text):
        msgBox = QMessageBox()
        msgBox.setIcon(QMessageBox.Information)
        msgBox.setText(text)        
        msgBox.setWindowTitle("Внимание!")
        msgBox.setFixedSize(300,150)
        msgBox.setStandardButtons(QMessageBox.Ok)
        msgBox.exec()

class Response():
    def __init__(self):
        self.method = None
        self.predicts_df = None
        self.val_predicts_df = None
        self.best_ma = None
        self.best_params = None
        self.test_y = None


class PredictionMethods(QtCore.QObject):
    finished = pyqtSignal()
    result = pyqtSignal(Response)
    def __init__(self,stationary_df,multiseries_df,singleseries_df,step_size,method,inverting_times,best_ma,best_params):
        super().__init__()
        self.df = stationary_df
        self.methods = ["VARMA","Boosting on decision trees","LSTM"]
        self.widget_method = method
        self.multiseries_df=multiseries_df
        self.singleseries_df=singleseries_df
        self.step_size = step_size
        self.inverting_times = inverting_times
        self.ma = best_ma
        self.params = best_params
        self.response = Response()

    def nstep_prediction(self):
        if self.widget_method == 'VARMA':
            multi_predicts_df = varma_prediction(self.multiseries_df,None,steps = self.step_size)
            single_predicts_df = arma_prediction(self.singleseries_df,None,steps = self.step_size)
            self.response.predicts_df = pd.concat([multi_predicts_df, single_predicts_df], axis=1)
            self.response.method = self.widget_method
            self.finished.emit() 
            self.result.emit(self.response)
        elif self.widget_method == "BoDT":
            self.response.predicts_df = boosting_prediction(self.df,self.df.columns,self.step_size,self.ma,self.params,self.inverting_times)
            self.response.method = self.widget_method
            self.finished.emit() 
            self.result.emit(self.response)
        elif self.widget_method == "LSTM":
            pass

class ValidPredictionMethods(QtCore.QObject):
    finished = pyqtSignal()
    result = pyqtSignal(Response)
    def __init__(self,stationary_df,multi_train_df,multi_valid_df,single_train_df,single_valid_df,method,inverting_times):
        super().__init__()
        self.df = stationary_df
        self.multi_train_df = multi_train_df
        self.multi_valid_df = multi_valid_df
        self.single_train_df = single_train_df
        self.single_valid_df = single_valid_df
        self.widget_method = method
        self.inverting_times = inverting_times
        self.response = Response()

    def validation_prediction(self):
        if self.widget_method == 'VARMA':
            multi_predicts_df = varma_prediction(self.multi_train_df,self.multi_valid_df,steps = None)
            single_predicts_df = arma_prediction(self.single_train_df,self.single_valid_df,steps = None)
            self.response.val_predicts_df = pd.concat([multi_predicts_df, single_predicts_df], axis=1)
            self.response.method = self.widget_method
            self.finished.emit() 
            self.result.emit(self.response)
        elif self.widget_method == "BoDT":
            self.response.val_predicts_df, test_y, best_ma, best_params = boosting_validation(self.df,self.df.columns,0.75,self.inverting_times)
            self.response.method = self.widget_method
            self.response.test_y = test_y
            self.response.best_ma = best_ma
            self.response.best_params = best_params
            self.finished.emit() 
            self.result.emit(self.response)
        elif self.widget_method == "LSTM":
            pass

class Window_predictions(QWidget):
    def __init__(self,df,stationary_df,inverting_times):
        super().__init__()
        self.df = df
        self.stationary_df = stationary_df
        self.inverting_times = inverting_times
        self.multiseries_df,self.singleseries_df = multi_single_ts(self.df)
        self.methods = ["VARMA","BoDT","LSTM"]
        self.setWindowTitle("VarPrediction")
        self.Button1 = QPushButton("Валидировать")
        self.Button2 = QPushButton("Предсказать на n шагов")
        self.val_textbox = QLineEdit(self)
        self.nstep_textbox = QLineEdit(self)
        self.listwidget = QListWidget()
        self.listwidget.setFixedSize(300,50)
        for i,method in enumerate(self.methods):
            self.listwidget.insertItem(i,method)
        self.val_label = QLabel('Введите размер валидационной выборки', self)
        self.nstep_label = QLabel('Введите кол-во предсказанных шагов', self)
        self.Button3=QPushButton("Результаты тестового прогноза")
        self.Button4=QPushButton("Результаты прогноза на n шагов")
        self.Button5=QPushButton("Сохранить результаты \nтестового прогноза")
        self.Button6=QPushButton("Сохранить результаты тестового \nпрогноза на n шагов")
        self.Button7=QPushButton("Графики прогнозов")
        self.method_label = QLabel('Выберите метод прогноза', self)
        self.Button1.setFixedSize(300, 50)
        self.Button2.setFixedSize(300, 50)
        self.Button3.setFixedSize(300, 50)
        self.Button4.setFixedSize(300, 50)
        self.Button5.setFixedSize(300, 50)
        self.Button6.setFixedSize(300, 50)
        self.Button7.setFixedSize(300, 70)
        self.val_textbox.setFixedSize(300, 30)
        self.nstep_textbox.setFixedSize(300, 30)
        self.val_label.setFixedSize(300,50)
        self.nstep_label.setFixedSize(300,50)
        self.listwidget.setFixedSize(300,70)
        layout = QGridLayout()
        layout.addWidget(self.method_label, 0, 0)
        layout.addWidget(self.listwidget, 1, 0)
        layout.addWidget(self.Button1, 4, 0)
        layout.addWidget(self.Button2, 4, 1)
        layout.addWidget(self.val_label, 2, 0)
        layout.addWidget(self.nstep_label, 2, 1)
        layout.addWidget(self.val_textbox, 3, 0)
        layout.addWidget(self.nstep_textbox, 3, 1)
        layout.addWidget(self.Button3, 5, 0)
        layout.addWidget(self.Button4, 5, 1)
        layout.addWidget(self.Button5, 6, 0)
        layout.addWidget(self.Button6, 6, 1)
        layout.addWidget(self.Button7, 1, 1)
        layout.setHorizontalSpacing(5)
        layout.setVerticalSpacing(5)
        self.Button1.clicked.connect(self.validation_prediction)
        self.Button2.clicked.connect(self.nstep_prediction)
        self.Button3.clicked.connect(self.table_validation)
        self.Button4.clicked.connect(self.table_nstep)
        self.Button5.clicked.connect(self.save_valid)
        self.Button6.clicked.connect(self.save_prediction)
        self.Button7.clicked.connect(self.plot_prediction)
        self.listwidget.clicked.connect(self.clicked_method)
        self.Button1.setFont(QtGui.QFont("Serif", 8, QtGui.QFont.Bold))
        self.Button2.setFont(QtGui.QFont("Serif", 8, QtGui.QFont.Bold))
        self.Button3.setFont(QtGui.QFont("Serif", 8, QtGui.QFont.Bold))
        self.Button4.setFont(QtGui.QFont("Serif", 8, QtGui.QFont.Bold))
        self.Button5.setFont(QtGui.QFont("Serif", 8, QtGui.QFont.Bold))
        self.Button6.setFont(QtGui.QFont("Serif", 8, QtGui.QFont.Bold))
        self.Button7.setFont(QtGui.QFont("Serif", 8, QtGui.QFont.Bold))
        self.listwidget.setFont(QtGui.QFont("Serif", 8, QtGui.QFont.Bold))
        self.val_label.setFont(QtGui.QFont("Serif", 8, QtGui.QFont.Bold))
        self.nstep_label.setFont(QtGui.QFont("Serif", 8, QtGui.QFont.Bold))
        self.method_label.setFont(QtGui.QFont("Serif", 8, QtGui.QFont.Bold))
        self.val_predicts_df = dict.fromkeys(self.methods, None)
        self.predicts_df = dict.fromkeys(self.methods, None)
        self.val_metrics_df = dict.fromkeys(self.methods, None)
        self.view1 = QTableView()
        self.view2 = QTableView()
        self.view3 = QTableView()
        self.view4 = QTableView()
        self.train_df = dict.fromkeys(self.methods, None)
        self.valid_df = dict.fromkeys(self.methods, None)
        self.widget_item = None
        self.BoDT_best_ma = None
        self.LSTM_best_ma = None
        self.BoDT_params = None
        self.w_graphs = Prediction_graphs(self.df,self.val_predicts_df,self.predicts_df)
        self.setLayout(layout)

    def plot_prediction(self):
        self.w_graphs.show()

    def validation_prediction(self):
        try:
            val_size = 1 - float(self.val_textbox.text())
        except:
            self.warning_box("Введите число от 0.05 до 0.5")
        try:
            if (val_size>=0.5 and val_size<=0.95):
                self.train_df['VARMA'], self.valid_df['VARMA'] = train_valid_split(self.stationary_df,val_size)
                multi_train_df, multi_valid_df = self.train_df['VARMA'][self.multiseries_df.columns],self.valid_df['VARMA'][self.multiseries_df.columns]
                single_train_df, single_valid_df = self.train_df['VARMA'][self.singleseries_df.columns],self.valid_df['VARMA'][self.singleseries_df.columns]
                if self.widget_item == 'VARMA':
                    self.thread = QtCore.QThread()
                    self.ValidPred = ValidPredictionMethods(self.stationary_df,multi_train_df,multi_valid_df,single_train_df,single_valid_df,self.widget_item,self.inverting_times)
                    self.ValidPred.moveToThread(self.thread)
                    self.thread.started.connect(self.ValidPred.validation_prediction)
                    self.ValidPred.finished.connect(self.thread.quit)
                    self.ValidPred.finished.connect(self.ValidPred.deleteLater)
                    self.thread.finished.connect(self.thread.deleteLater)
                    self.thread.start()
                    self.ValidPred.result.connect(self.get_valid_method_output)
                    self.Button1.setEnabled(False)
                    self.Button2.setEnabled(False)
                    self.thread.finished.connect(lambda: self.enableButton())
                elif self.widget_item == 'BoDT':
                    self.thread = QtCore.QThread()
                    self.ValidPred = ValidPredictionMethods(self.stationary_df,multi_train_df,multi_valid_df,single_train_df,single_valid_df,self.widget_item,self.inverting_times)
                    self.ValidPred.moveToThread(self.thread)
                    self.thread.started.connect(self.ValidPred.validation_prediction)
                    self.ValidPred.finished.connect(self.thread.quit)
                    self.ValidPred.finished.connect(self.ValidPred.deleteLater)
                    self.thread.finished.connect(self.thread.deleteLater)
                    self.thread.start()
                    self.ValidPred.result.connect(self.get_valid_method_output)
                    self.Button1.setEnabled(False)
                    self.Button2.setEnabled(False)
                    self.thread.finished.connect(lambda: self.enableButton())
                elif self.widget_item == 'LSTM':
                    pass
            else:
                self.warning_box("Введите число от 0.05 до 0.5")
        except Exception as e:
            self.warning_box(f"Ошибка {e}")

    def nstep_prediction(self):
        try:
            try:
                step_size = int(self.nstep_textbox.text())
            except:
                self.warning_box("Введите число от 1 до 20")
            if (step_size>0 and step_size<=20):
                if self.widget_item == 'VARMA':
                    self.thread = QtCore.QThread()
                    self.Pred = PredictionMethods(self.stationary_df,self.multiseries_df,self.singleseries_df,step_size,self.widget_item,self.inverting_times,self.BoDT_best_ma,self.BoDT_params)
                    self.Pred.moveToThread(self.thread)
                    self.thread.started.connect(self.Pred.nstep_prediction)
                    self.Pred.finished.connect(self.thread.quit)
                    self.Pred.finished.connect(self.Pred.deleteLater)
                    self.thread.finished.connect(self.thread.deleteLater)
                    self.thread.start()
                    self.Pred.result.connect(self.get_predict_method_output)
                    self.Button1.setEnabled(False)
                    self.Button2.setEnabled(False)
                    self.thread.finished.connect(lambda: self.enableButton())
                elif self.widget_item == 'BoDT':
                    if self.BoDT_best_ma is None and self.BoDT_params is None:
                        self.warning_box("Сперва cделайте валидацию!")
                    else:
                        self.thread = QtCore.QThread()
                        self.Pred = PredictionMethods(self.stationary_df,self.multiseries_df,self.singleseries_df,step_size,self.widget_item,self.inverting_times,self.BoDT_best_ma,self.BoDT_params)
                        self.Pred.moveToThread(self.thread)
                        self.thread.started.connect(self.Pred.nstep_prediction)
                        self.Pred.finished.connect(self.thread.quit)
                        self.Pred.finished.connect(self.Pred.deleteLater)
                        self.thread.finished.connect(self.thread.deleteLater)
                        self.thread.start()
                        self.Pred.result.connect(self.get_predict_method_output)
                        self.Button1.setEnabled(False)
                        self.Button2.setEnabled(False)
                        self.thread.finished.connect(lambda: self.enableButton())
                elif self.widget_item == 'LSTM':
                    pass
            else:
                self.warning_box("Введите число от 1 до 20")
        except Exception as e:
           print(e)

    def enableButton(self,mode = True):
        self.Button1.setEnabled(mode)
        self.Button2.setEnabled(mode)

    def get_valid_method_output(self,s):
        method = s.method
        self.BoDT_best_ma = s.best_ma
        self.BoDT_params = s.best_params
        self.val_predicts_df[method] = s.val_predicts_df
        if method=='BoDT' or method=='LSTM':
            self.valid_df[method] = s.test_y
        self.val_metrics_df[method] = metrics(self.valid_df[method],self.val_predicts_df[method])
        self.val_predicts_df[method] = invert_diff(self.val_predicts_df[method],self.inverting_times)
    
    def get_predict_method_output(self,s):
        method = s.method
        self.predicts_df[method] = s.predicts_df
        self.predicts_df[method] = invert_diff(self.predicts_df[method],self.inverting_times)

    def save_valid(self):
        for key in self.val_predicts_df.keys():
            if self.val_predicts_df[key] is not None:
                self.val_predicts_df[key].to_csv(f'output/val_{key}_predicts.csv',index=False)
        for key in self.val_metrics_df.keys():
            if self.val_metrics_df[key] is not None:
                self.val_metrics_df[key].to_csv(f'output/{key}_metrics.csv',index=False)        

    def save_prediction(self):
        for key in self.predicts_df.keys():
            if self.predicts_df[key] is not None:
                self.predicts_df[key].to_csv(f'output/{key}_predicts.csv',index=False)

    def plot_valid(self):
        pass

    def plot_predictions(self):
        pass

    def clicked_method(self):
        item = self.listwidget.currentItem()
        self.widget_item = item.text()

    def table_validation(self):
        if self.val_metrics_df[self.widget_item] is not None and self.val_predicts_df[self.widget_item] is not None :
            self.show_tables(self.view1,self.val_predicts_df[self.widget_item])
            self.show_tables(self.view2,self.val_metrics_df[self.widget_item])
        else:
            self.warning_box("Сперва нажмите на кнопку\n'Валидировать'")

    def table_nstep(self):
        if self.predicts_df[self.widget_item] is not None:
            self.show_tables(self.view3,self.predicts_df[self.widget_item])
        else:
            self.warning_box("Сперва нажмите на кнопку\n'Предсказать на n шагов'")
    
    def warning_box(self,text):
        msgBox = QMessageBox()
        msgBox.setIcon(QMessageBox.Information)
        msgBox.setText(text)        
        msgBox.setWindowTitle("Внимание!")
        msgBox.setFixedSize(300,150)
        msgBox.setStandardButtons(QMessageBox.Ok)
        msgBox.exec()

    def show_tables(self,view,df):
        df = pandasModel(df)
        view.setModel(df)
        view.move(600, 500)
        view.resize(800,600)
        view.resizeColumnsToContents()
        view.show()

class pandasModel(QAbstractTableModel):
    def __init__(self, data):
        QAbstractTableModel.__init__(self)
        self._data = data

    def rowCount(self, parent=None):
        return self._data.shape[0]

    def columnCount(self, parnet=None):
        return self._data.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid():
            if role == Qt.DisplayRole:
                tmp = self._data.iloc[index.row(), index.column()]
                if isinstance(tmp, float):
                    return "%.2f" % tmp
                return str(tmp)
        return None

    def headerData(self, col, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self._data.columns[col]
        elif orientation == Qt.Vertical and role == Qt.DisplayRole:
            return self._data.index[col]
        return None