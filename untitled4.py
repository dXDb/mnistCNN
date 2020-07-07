# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 19:17:52 2020

@author: CVPR
"""
from PyQt5.QtWidgets         import *
from PyQt5.QtCore            import *
from PyQt5.QtGui             import *
from tensorflow              import keras
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

import cv2
import sys
import time
import numpy as np
import tensorflow as tf

class MyCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        if time.time() - self.start_time > 30:
            self.model.stop_training = True


class MnistGUI(QMainWindow):
    def __init__(self, parent = None):
        super(MnistGUI, self).__init__(parent)
        # 초기 모델
        self.model = keras.Sequential()
        self.model.add(Flatten(input_shape=(28, 28, 1)))
        self.model.add(Dense(128, activation = 'sigmoid'))
        self.model.add(Dense(10, activation = 'softmax'))
            
        # Main
        self.setWindowTitle("류원정짱")

        # Widget 생성
        wid                = QWidget(self)
        self.canvasLabel   = QLabel()
        self.canvas        = QPixmap(280, 280)
        self.predictButton = QPushButton()
        self.resetButton   = QPushButton()
        self.mnist1        = QRadioButton()
        self.mnist2        = QRadioButton()
        self.mnist3        = QRadioButton()
        self.mnist4        = QRadioButton()
        self.mnist5        = QRadioButton()

        # Widget text 설정
        self.predictButton.setText("predict")
        self.resetButton.setText("reset")
        self.mnist1.setText("1번째")
        self.mnist2.setText("2번째")
        self.mnist3.setText("3번째")
        self.mnist4.setText("4번째")
        self.mnist5.setText("5번째")
        self.mnist1.setChecked(True)

        # Widget 기능 설정
        self.predictButton.clicked.connect(self.predict)
        self.resetButton.clicked.connect(self.reset)
        self.mnist1.clicked.connect(self.radioButtonClicked)
        self.mnist2.clicked.connect(self.radioButtonClicked)
        self.mnist3.clicked.connect(self.radioButtonClicked)
        self.mnist4.clicked.connect(self.radioButtonClicked)
        self.mnist5.clicked.connect(self.radioButtonClicked)

        # Widget 크기 설정
        self.mnist1.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.mnist2.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.mnist3.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.mnist4.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.mnist5.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.predictButton.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        self.resetButton.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)

        # canvas 지정
        self.canvasLabel.setPixmap(self.canvas)
        self.last_x, self.last_y = None, None
        self.reset()

        # Widget 위치 설정
        self.setCentralWidget(wid)
        buttonLayout = QHBoxLayout()
        buttonLayout.addWidget(self.predictButton)
        buttonLayout.addWidget(self.resetButton)

        radioLayout = QVBoxLayout()
        radioLayout.addWidget(self.mnist1)
        radioLayout.addWidget(self.mnist2)
        radioLayout.addWidget(self.mnist3)
        radioLayout.addWidget(self.mnist4)
        radioLayout.addWidget(self.mnist5)

        controlLayout = QVBoxLayout()
        controlLayout.addLayout(radioLayout)
        controlLayout.addLayout(buttonLayout)

        WholeLayout = QHBoxLayout()
        WholeLayout.addWidget(self.canvasLabel)
        WholeLayout.addLayout(controlLayout)
        
        wid.setLayout(WholeLayout)
            
    
    # 종료
    def exitCall(self):
        sys.exit(app.exec_())

    # Mnist 그리기
    def mouseMoveEvent(self, e):
        if self.last_x is None:
            self.last_x = e.x()
            self.last_y = e.y()
            return

        painter = QPainter(self.canvasLabel.pixmap())
        p = painter.pen()
        p.setWidth(20)
        p.setColor(QColor('#FFFFFF'))
        painter.setPen(p)
        painter.drawLine(self.last_x - 10, self.last_y - 10, e.x() - 10, e.y() - 10)
        painter.end()
        self.update()

        self.last_x = e.x()
        self.last_y = e.y()

    def mouseReleaseEvent(self, e):
        self.last_x = None
        self.last_y = None

    def radioButtonClicked(self):
        if self.mnist1.isChecked():
            self.model = keras.Sequential()
            self.model.add(Flatten(input_shape=(28, 28, 1)))
            self.model.add(Dense(128, activation = 'sigmoid'))
            self.model.add(Dense(10, activation = 'softmax'))

        elif self.mnist2.isChecked():
            self.model = keras.Sequential()
            self.model.add(Conv2D(32, (3, 3), activation = 'relu', input_shape=(28, 28, 1)))
            self.model.add(MaxPooling2D(pool_size=(2, 2)))
            self.model.add(Flatten())
            self.model.add(Dense(128, activation = 'sigmoid'))
            self.model.add(Dense(10, activation = 'softmax'))

        elif self.mnist3.isChecked():
            self.model = keras.Sequential()
            self.model.add(Conv2D(32, (3, 3), activation = 'relu', input_shape=(28, 28, 1)))
            self.model.add(Conv2D(64, (3, 3), activation = 'relu'))
            self.model.add(MaxPooling2D(pool_size=(2, 2)))
            self.model.add(Flatten())
            self.model.add(Dense(128, activation = 'sigmoid'))
            self.model.add(Dense(10, activation = 'softmax'))

        elif self.mnist4.isChecked():
            self.model = keras.Sequential()
            self.model.add(Conv2D(32, (3, 3), activation = 'relu', input_shape=(28, 28, 1)))
            self.model.add(MaxPooling2D(pool_size=(2, 2)))
            self.model.add(Conv2D(64, (3, 3), activation = 'relu'))
            self.model.add(MaxPooling2D(pool_size=(2, 2)))
            self.model.add(Flatten())
            self.model.add(Dense(128, activation = 'sigmoid'))
            self.model.add(Dense(10, activation = 'softmax'))

        elif self.mnist5.isChecked():
            self.model = keras.Sequential()
            self.model.add(Conv2D(32, (3, 3), activation = 'relu', input_shape=(28, 28, 1)))
            self.model.add(MaxPooling2D(pool_size=(2, 2)))
            self.model.add(Conv2D(64, (3, 3), activation = 'relu'))
            self.model.add(MaxPooling2D(pool_size=(2, 2)))
            self.model.add(Flatten())
            self.model.add(Dense(64, activation = 'sigmoid'))
            self.model.add(Dense(32, activation = 'sigmoid'))
            self.model.add(Dense(10, activation = 'softmax'))

    def predict(self):
        self.canvasLabel.pixmap().save("SavedImage.jpg")
        testData = cv2.imread("SavedImage.jpg", cv2.IMREAD_GRAYSCALE)
        testData = cv2.resize(testData, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)

        earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', min_delta=0.0, patience=3)

        mnist = tf.keras.datasets.mnist
        (x_train, y_train_origin), (x_test, y_test_origin) = mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0           # 0과 1 사이의 값으로 변환

        testData = testData / 255.0
        testData = testData.reshape((1, 28, 28, 1))
        
        x_train = x_train.reshape((-1, 28, 28, 1))
        x_test = x_test.reshape((-1, 28, 28, 1))

        nb_classes = 10
        y_train = keras.utils.to_categorical(y_train_origin, num_classes = nb_classes)       # one-hot encoding
        y_test = keras.utils.to_categorical(y_test_origin, num_classes = nb_classes)       # one-hot encoding

        self.model.compile(optimizer = keras.optimizers.Adam(learning_rate = 0.01), loss = 'categorical_crossentropy', metrics = ['categorical_accuracy'])
        hist = self.model.fit(x_train, y_train, epochs=10000, batch_size=1000, validation_data=(x_test, y_test), callbacks=[earlystop_callback, MyCallback()])
        self.model.evaluate(x_test, y_test)

        y_new = self.model.predict(testData)
        
        
        QMessageBox.question(self, '류원정 짱', "{}".format(y_new.argmax()), QMessageBox.Cancel, QMessageBox.Cancel)


    def reset(self):
        painter = QPainter(self.canvasLabel.pixmap())
        painter.setBrush(QColor(0, 0, 0))
        painter.drawRect(0, 0, 280, 280)
        painter.end()
        self.update()


# 메인 함수
if __name__  == "__main__":
    # GUI 생성
    app = QApplication(sys.argv)

    pannel = MnistGUI()
    pannel.resize(480, 280)
    pannel.show()
    sys.exit(app.exec_())
