import numpy
import os, sys
import tensorflow as tf
from threading import Thread
import numpy as np
from random import randint, random
from PyQt5 import QtWidgets, QtGui, QtCore
from cvae_sampler import *
from util import *

class RobosoccerGrid(QtWidgets.QGraphicsItem) :
    def __init__(self, parent=None, *args, **kwargs):
        QtWidgets.QGraphicsItem.__init__(self,parent)
        self.width = 2200
        self.height = 1400
        self.cell_size = 50
        self.cells = []
        cs = self.cell_size
        c = QtCore.QRectF(-cs/2,-cs/2,cs,cs)
        for i in range(self.width/self.cell_size) :
            for j in range(self.height/self.cell_size) :
                cell = c.translated(i*cs-self.width/2,j*cs-self.height/2)
                self.cells.append(cell)
        self.samples_pos = []
        self.start_pos = None
        self.goal_pos = None
        self.grid_map = None
        self.circles = []
        self.occ_cells = []
        self.radius = 50.0
        self.x_start = [0.0, 0.0, 0.0, 0.0]
        self.x_goal = [0.0, 0.0, 0.0, 0.0]

    def setStates(self, start, goal) :
        self.start_pos = QtCore.QPointF(start[0]*100.0, start[1]*100.0)
        self.goal_pos = QtCore.QPointF(goal[0]*100.0, goal[1]*100.0)
    
    def setSamples(self, samples) :
        self.samples_pos = []
        self.samples_pos = [QtCore.QPointF(s[0]*100.0, s[1]*100.0) for s in samples]
        # print samples
        # print self.samples_pos

    def fromGridMap(self, grid) :
        self.grid_map = grid

    def setCircles(self, circles) :
        self.circles = [QtCore.QPointF(c[0]*100.0, c[1]*100.0) for c in circles]

    def setOccupied(self, obs, radius) :
        circles = occupied_cells(obs, radius, self.cell_size)
        self.radius = radius
        self.occ_cells = self.to_qt_rect(circles)

    def to_qt_rect(self, circles) :
        qt_rect = []
        cs = self.cell_size
        c0 = QtCore.QRectF(-cs/2,-cs/2,cs,cs)
        for c in circles :
            qt_rect.append(c0.translated(c[0]*cs,c[1]*cs))
        return qt_rect

    def paint(self, painter, option, widget=None) :
        p = QtGui.QPainter()
        color = QtGui.QColor('blue')
        painter.setPen(color)
        painter.drawRects(self.cells)
        color = QtGui.QColor('black')
        painter.setPen(color)
        painter.setBrush(color)
        # painter.drawRects(self.occ_cells)
        r = QtCore.QRectF(-self.cell_size/2,-self.cell_size/2,self.cell_size,self.cell_size)
        if not (self.grid_map is None) :
            s = self.grid_map.shape
            for i in range(s[0]) :
                for j in range(s[1]) :
                    if self.grid_map[i][j] > 0.0 :
                        painter.drawRect(r.translated(i*self.cell_size-self.width/2,j*self.cell_size-self.height/2))
        color = QtGui.QColor('gray')
        painter.setPen(color)
        painter.setBrush(color)
        #draw origin
        painter.drawEllipse(0,0,10,10)
        color = QtGui.QColor('green')
        painter.setPen(color)
        painter.setBrush(color)
        for s in self.samples_pos :
            painter.drawEllipse(s, 10.0, 10.0)
        if not (self.start_pos is None) :
            color = QtGui.QColor('red')
            painter.setPen(color)
            painter.setBrush(color)
            painter.drawEllipse(self.start_pos, 10.0, 10.0)
        if not (self.goal_pos is None) :
            color = QtGui.QColor('blue')
            painter.setPen(color)
            painter.setBrush(color)
            painter.drawEllipse(self.goal_pos, 10.0, 10.0)
        for c in self.circles :
            color = QtGui.QColor('gray')
            painter.setPen(color)
            painter.setBrush(color)
            r = self.radius
            painter.drawEllipse(c,r*2,r*2)

    def boundingRect(self) :
        w = self.width
        h = self.height
        return QtCore.QRectF(-w/2,-h/2,w,h)

class Window(QtWidgets.QWidget) :
    def __init__(self, *args, **kwargs):
        # self.widget = QtWidgets.QWidget()
        QtWidgets.QWidget.__init__(self)
        self.graphics_view = QtWidgets.QGraphicsView()
        self.line_edit = QtWidgets.QLineEdit()
        self.it_spin = QtWidgets.QSpinBox()
        self.train_btn = QtWidgets.QPushButton("Train!")
        self.test_btn = QtWidgets.QPushButton("Random Test")
        self.open_btn = QtWidgets.QPushButton("Open")
        self.next_btn = QtWidgets.QPushButton(">")
        self.prev_btn = QtWidgets.QPushButton("<")
        self.next_btn.setEnabled(False)
        self.prev_btn.setEnabled(False)
        self.it_spin.setEnabled(False)
        self.field = RobosoccerGrid()
        grid_layout = QtWidgets.QGridLayout()
        line_layout = QtWidgets.QGridLayout()
        line_layout.addWidget(self.prev_btn,0,0)
        line_layout.addWidget(self.next_btn,0,1)
        line_layout.addWidget(self.it_spin,0,2)
        line_layout.addWidget(self.line_edit,0,3)
        line_layout.addWidget(self.open_btn,0,4)
        line_layout.addWidget(self.train_btn,0,5)
        line_layout.addWidget(self.test_btn,0,6)
        grid_layout.addLayout(line_layout,0,0)
        grid_layout.addWidget(self.graphics_view,1,0)
        # self.widget.setLayout(grid_layout)
        self.setLayout(grid_layout)

        self.open_btn.clicked.connect(self.open)
        self.train_btn.clicked.connect(self.train)
        self.test_btn.clicked.connect(self.test)
        self.next_btn.clicked.connect(self.nextData)
        self.prev_btn.clicked.connect(self.prevData)
        self.it_spin.valueChanged.connect(self.jumpData)

        self.scene = QtWidgets.QGraphicsScene(self.field.boundingRect())
        self.scene.addItem(self.field)
        self.graphics_view.setScene(self.scene)
        self.graphics_view.scale(0.4,0.4)

        if 'dataset' in kwargs :
            self.dataset = kwargs['dataset']
        else :
            self.dataset = Dataset()
        if 'sampler' in kwargs :
            self.cvae_sampler = kwargs['sampler']
        else :
            self.cvae_sampler = CVAESampler()

        if not (self.dataset.abspath is '') :
            self.line_edit.setText(self.dataset.abspath)
            self.next_btn.setEnabled(True)
            self.prev_btn.setEnabled(True)
            self.it_spin.setEnabled(True)
            n = self.dataset.n_data
            self.it_spin.setMaximum(n-1)

        self.it = 0
        self.ok = True

        self.width = kwargs['width'] if 'width' in kwargs else WIDTH
        self.height = kwargs['height'] if 'height' in kwargs else HEIGHT
        self.cell = kwargs['cell'] if 'cell' in kwargs else CELL

    def closeEvent(self, event) :
        self.ok = False

    def train_loop(self) :
        if self.dataset.mode == 'load_all' :
            dataset = self.dataset.get_data(0.8)
            train_data = dataset['train']
            x = train_data['samples']
            c = train_data['conditions']
            print 'train with all data loaded to memory'
            while self.ok :
                self.cvae_sampler.it += 1
                x_mb, c_mb = self.cvae_sampler.shuffled_data(256, x, c)
                self.cvae_sampler.train_step(x_mb, c_mb)
        elif self.dataset.mode == 'load_partial' :
            print 'train with partial data loading'
            strings = self.dataset.str
            while self.ok :
                self.cvae_sampler.it += 1
                x_mb, c_mb = self.cvae_sampler.shuffled_data_from_str(256, strings, len(strings))
                self.cvae_sampler.train_step(x_mb, c_mb)
        self.cvae_sampler.save()

    def test(self) :
        x_scale, x_shift = self.width/100.0, self.width/200.0
        y_scale, y_shift = self.height/100.0, self.height/200.0
        v_scale, v_shift = 2*1.5, 1.5/2
        rg = np.random.random
        xs = np.reshape([rg(1)*x_scale-x_shift, rg(1)*y_scale-y_shift, rg(1)*v_scale-v_shift, rg(1)*v_scale-v_shift],(4))
        xg = np.reshape([rg(1)*x_scale-x_shift, rg(1)*y_scale-y_shift, rg(1)*v_scale-v_shift, rg(1)*v_scale-v_shift],(4))
        obs = np.reshape([[rg(1)*x_scale-x_shift,rg(1)*y_scale-y_shift] for _ in range(9)],(9,2))
        y, z = self.cvae_sampler.sample(30, xs, xg, obs)
        self.drawData([xs, xg, obs, y])
        self.scene.update()

    def train(self) :
        if self.train_btn.text() == 'Train!' :
            self.ok = True
            thread = Thread(target=self.train_loop)
            print 'starting train_loop thread'
            thread.start()
            # check if self exist
            if self : self.train_btn.setText('Stop!')
        else :
            self.ok = False
            self.train_btn.setText('Train!')

    def jumpData(self) :
        self.it = self.it_spin.value()
        data = self.dataset.get(self.it)
        self.drawData(data)
        self.scene.update()

    def nextData(self) :
        self.it = self.it+1
        data = self.dataset.get(self.it)
        self.drawData(data)
        self.scene.update()

    def prevData(self) :
        self.it = self.it-1
        data = self.dataset.get(self.it)
        self.drawData(data)
        self.scene.update()

    def drawData(self, data) :
        r = self.cell
        w = self.width
        h = self.height
        c = self.cell
        self.field.setCircles(data[2])
        self.field.fromGridMap(grid_map(data[2],r,w,h,c))
        self.field.setStates(data[0],data[1])
        self.field.setSamples(data[3])

    def open(self) :
        filename = QtWidgets.QFileDialog.getOpenFileName()[0]
        self.line_edit.setText(filename)
        print 'opening.. : ', filename
        n = self.dataset.load(filename)
        print 'loaded', n, 'data'
        data = self.dataset.get(1)
        self.drawData(data)
        self.next_btn.setEnabled(True)
        self.prev_btn.setEnabled(True)
        self.it_spin.setEnabled(True)
        self.it_spin.setMaximum(n-1)