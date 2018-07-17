import numpy
import os, sys
import tensorflow as tf
from random import randint, random
from PyQt5 import QtWidgets, QtGui, QtCore
from cvae_sampler import *
from gui import *
from util import *

if __name__ == '__main__' :
    app = QtWidgets.QApplication(sys.argv)
    dataset = Dataset()
    cvae = CVAESampler(epoch=2000000)
    dataset.load('solutions.txt')
    window = Window(dataset=dataset, sampler=cvae)
    window.show()
    app.exec_()
