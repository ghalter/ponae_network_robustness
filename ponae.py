import numpy as np
import typing
from random import randint, shuffle, random

from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

import networkx as nx

class MainWindow(QMainWindow):
    def __init__(self, parent):
        super(MainWindow, self).__init__(parent=parent)
        self.setWindowTitle("Only One Node Away from Extinction")
        self.robustness = MatplotlibWidget(self)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.robustness)

        self.controls = Controls(self)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.controls)
        self.controls.onRandomnessChanged.connect(self.compute_robustness)
        self.controls.onIterationChanged.connect(self.view_iteration)
        self.controls.onGraphChanged.connect(self.create_graph)

        self.view = NetworkView(self)
        self.setCentralWidget(self.view)

        self.graph = nx.generators.random_graphs.barabasi_albert_graph(300, 2)
        self.max_edges = 0
        self.update_graph()
        self.view.draw_graph(self.graph, re_layout=True)

        self.graph_cache = []
        self.n_simulations = 2
        self.show()

    @pyqtSlot(int, int, int, str)
    def create_graph(self, n_nodes, connectivity, n_simulations, mode="Barabasi Albert Graph"):
        self.n_simulations = n_simulations
        if mode == "Barabasi Albert Graph":
            self.graph = nx.generators.random_graphs.barabasi_albert_graph(n_nodes, connectivity)
        elif mode == "Regular":
            self.graph = nx.generators.random_graphs.random_regular_graph(connectivity, n_nodes)
        elif mode == "Erdös-Reny Graph":
            self.graph = nx.generators.random_graphs.erdos_renyi_graph(n_nodes, connectivity / 10)
        self.update_graph()
        self.view.draw_graph(self.graph, re_layout=True)

    def update_graph(self, g = None):
        if g is None:
            g = self.graph
        self.max_edges = 0
        for v in g.nodes:
            n = len(g.edges(v))
            if n > self.max_edges:
                self.max_edges = n

    def view_iteration(self, it):
        if it == 0:
            self.update_graph(self.graph)
            self.view.draw_graph(self.graph, re_layout=False)
        elif it < len(self.graph_cache):
            # self.update_graph(self.graph_cache[it])
            self.view.draw_graph(self.graph_cache[it], re_layout=False)

    def compute_robustness(self, r):
        self.graph_cache = []
        bias = r * 2.0 - 1.0


        t_max_nodes = []
        t_mean_nodes = []
        for step in range(self.n_simulations):
            g = self.graph.copy()
            max_nodes = []
            mean_nodes = []
            n_edges = []
            for i in range(len(self.graph.nodes)):
                to_remove = sorted([(idx, len(g.edges(idx))) for idx in g.nodes], key=lambda x: x[1], reverse=True)
                to_remove_sorted = [t[0] for t in to_remove]
                to_remove_random = [t[0] for t in to_remove]
                shuffle(to_remove_random)

                component_sizes = [len(s) for s in nx.connected_components(g)]
                max_nodes.append(np.amax(component_sizes))
                mean_nodes.append(np.mean(component_sizes))

                idx = randint(0, len(to_remove_sorted) - 1)
                if random() - bias < 0.0:
                    node = to_remove_random[idx]
                else:
                    node = to_remove_sorted[0]
                    idx = 0
                if node not in g.nodes:
                    continue
                g.remove_node(node)
                n_edges.append(len(g.edges))
                if step == 0:
                    self.graph_cache.append(g.copy())

            t_max_nodes.append(max_nodes)
            t_mean_nodes.append(mean_nodes)

        max_nodes = np.mean(t_max_nodes, axis=0)
        mean_nodes = np.mean(t_mean_nodes, axis=0)

        to_plot = []
        cropped_mean, cropped_max, xs = [], [], []
        for i in range(len(max_nodes)):
            if max_nodes[i] >= 1 or mean_nodes[i] >= 1:
                cropped_mean.append(mean_nodes[i])
                cropped_max.append(max_nodes[i])
                xs.append(i / len(max_nodes))

        self.controls.iteration_slider.setRange(0, len(self.graph_cache))
        self.robustness.plot(xs,
                             dict(mean_nodes=(xs, cropped_mean),
                                  max_nodes=(xs,cropped_max),
                                  n_edges=(list(range(len(n_edges))), n_edges)))


class NetworkView(QGraphicsView):
    R = 10
    SCALE = 1000

    def __init__(self, parent:MainWindow):
        super(NetworkView, self).__init__(parent)
        self.setScene(QGraphicsScene())
        self.setRenderHint(QPainter.Antialiasing)
        self.setMouseTracking(True)
        self.mw = parent

        self.items = dict()
        self.edges = []
        self.setBackgroundBrush(QColor(255,255,255))
        self.colormap = None

        self._is_creating_edge = False
        self.new_edge = None
        self.new_edge_n1 = None
        self.info_label = QLabel(self)
        self.info_label.move(10, 10)
        self.info_label.setStyleSheet("QLabel{background: transparent; color: rgb(10,10,10); font-size: 14pt;}")


    def draw_graph(self, g: nx.Graph, re_layout=True):
        self.colormap = cm.get_cmap("viridis", self.mw.max_edges)
        if re_layout:
            self.scene().clear()
            self.items = dict()
            self.edges = []
            lt = nx.spring_layout(g, iterations=50)

            coords = np.zeros(shape=(len(lt.keys()), 2))
            for i, (k, v) in enumerate(lt.items()):
                coords[i] = v

            coords -= np.amin(coords)
            coords /= np.amax(coords)
            coords *= self.SCALE

            for i in range(coords.shape[0]):
                v = coords[i]
                itm = NodeItem(self, i, v[0] - self.R, v[1] - self.R, 2*self.R, 2*self.R, _inhibit_update=True)
                itm.setZValue(100)
                self.scene().addItem(itm)
                self.items[i] = dict(item=itm, coords = v)
        else:
            for k,v in self.items.items():
                if k in g.nodes:
                    self.items[k]['item'].show()
                else:
                    self.items[k]['item'].hide()


        for k, v in self.items.items():
            v['item']._inhibit_update = False


        self.update_edges()
        # for n1, n2, w in g.edges:
        #     n1 = self.items[n1]['coords']
        #     n2 = self.items[n2]['coords']
        #
        #     self.scene().addLine(n1[0], n1[1], n2[0], n2[1], QColor(180,32,15))

        self.info_label.setText("Nodes {n}, Edges {e}".format(n=len(g.nodes), e=len(g.edges)))
        self.setSceneRect(0,0,self.SCALE, self.SCALE)

    @pyqtSlot()
    def update_edges(self):
        for e in self.edges:
            self.scene().removeItem(e)
        self.edges = []

        for q in self.mw.graph.edges:

            n1 = q[0]
            n2 = q[1]
            if n1 not in self.items or n2 not in self.items:
                continue
            n1 = self.items[n1]['item']
            n2 = self.items[n2]['item']
            if n1.isVisible() and n2.isVisible():
                n1 = (n1.get_center().x() + self.R, n1.get_center().y() + self.R)
                n2 = (n2.get_center().x() + self.R, n2.get_center().y() + self.R)
                # p1 = n1.mapFromItem(n1, QPointF())
                # p2 = n2.mapFromItem(n2, QPointF())
                # n1 = p1.x(), p1.y()
                # n2 = p2.x(), p2.y()
                itm = self.scene().addLine(n1[0], n1[1], n2[0], n2[1], QColor(180,32,15))
                itm.setZValue(0)
                self.edges.append(itm)

    def create_edge(self, n1):
        self.new_edge_n1 = n1
        self._is_creating_edge = True

    def finalize_edge(self, n2):
        self._is_creating_edge = False
        if self.new_edge_n1 is not n2:
            self.mw.graph.add_edge(self.new_edge_n1.idx, n2.idx)
        self.new_edge_n1 = None
        if self.new_edge is not None:
            self.scene().removeItem(self.new_edge)
        self.new_edge = None
        self.update_edges()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        super(NetworkView, self).mousePressEvent(event)
        if event.button() == Qt.RightButton:
            if self._is_creating_edge is False:
                idx = len(self.mw.graph.nodes.keys())
                self.mw.graph.add_node(idx)
                p = self.mapToScene(event.pos())
                v = p.x(),p.y()
                itm = NodeItem(self, idx, v[0] - self.R, v[1] - self.R, 2 * self.R, 2 * self.R, _inhibit_update=False)
                itm.setZValue(0)
                self.scene().addItem(itm)
                self.items[idx] = dict(item=itm, coords=v)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key_Escape:
            if self._is_creating_edge:
                self.finalize_edge(self.new_edge_n1)

    def resizeEvent(self, event: QResizeEvent) -> None:
        super(NetworkView, self).resizeEvent(event)
        self.setSceneRect(0, 0, self.SCALE, self.SCALE)
        self.fitInView(self.scene().itemsBoundingRect().adjusted(-40,-40,80,80), Qt.KeepAspectRatio)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        super(NetworkView, self).mouseMoveEvent(event)
        if self._is_creating_edge:
            if self.new_edge is not None:
                self.scene().removeItem(self.new_edge)
            s = self.new_edge_n1.get_center()
            x1 = s.x()
            y1 = s.y()

            s = self.mapFromGlobal(QCursor.pos())
            s = self.mapToScene(s)
            x2 = s.x()
            y2 = s.y()
            self.new_edge = self.scene().addLine(x1, y1, x2, y2)


class NodeItem(QGraphicsEllipseItem):

    def __init__(self, view:NetworkView, idx, x, y, w, h, _inhibit_update=False):
        self._inhibit_update = _inhibit_update
        super(NodeItem, self).__init__(x, y, w, h)
        self.idx = idx
        self._x = x
        self._y = y
        self.view = view
        self.setFlag(QGraphicsItem.ItemIsMovable)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges)
        self.edges = []
        self.last_pos = self.get_center()

    def boundingRect(self) -> QRectF:
        return super(NodeItem, self).boundingRect()

    def paint(self, painter: QPainter, option: 'QStyleOptionGraphicsItem', widget: typing.Optional[QWidget] = ...) -> None:
        p = QPen()
        value = len(self.view.mw.graph.edges(self.idx))/ np.clip(self.view.mw.max_edges, 1, None)
        c = self.view.colormap(value)
        self.setZValue(int(100 * value))
        alpha = 0.5 + (value / 0.5)
        p.setColor(QColor(180,32,15))
        p.setBrush(QColor(180,180,180))
        p.setWidth(5)
        # painter.setBrush(QColor(230,230,230))
        painter.setPen(p)
        p = QPainterPath()
        p.addEllipse(self.boundingRect())
        painter.drawPath(p)
        painter.fillPath(p, QColor(int(c[0] * 255),int(c[1] * 255),int(c[2] * 255)))

        # super(NodeItem, self).paint(painter, option, widget)

    def itemChange(self, change: 'QGraphicsItem.GraphicsItemChange', value: typing.Any) -> typing.Any:
        if not self._inhibit_update:
            if hasattr(self, "last_pos"):
                if self.last_pos != self.get_center():
                    self.view.update_edges()
                    self.last_pos = self.get_center()
            else:
                self.view.update_edges()
        return super(NodeItem, self).itemChange(change, value)

    def get_center(self):
        return self.mapToScene( self._x, self._y)

    def mousePressEvent(self, event: 'QGraphicsSceneMouseEvent') -> None:
        if event.button() == Qt.RightButton:
            self.view.create_edge(self)

        elif self.view._is_creating_edge:
            self.view.finalize_edge(self)

        super(NodeItem, self).mousePressEvent(event)


class Controls(QDockWidget):
    onRandomnessChanged = pyqtSignal(float)
    onIterationChanged = pyqtSignal(int)
    onGraphChanged = pyqtSignal(int, int, int, str)

    def __init__(self, parent):
        super(Controls, self).__init__(parent)
        self.inner = QWidget(self)
        lt = QGridLayout()
        self.inner.setLayout(lt)
        self.randomness_slider = QSlider( Qt.Horizontal, self)
        self.randomness_slider.setRange(0, 100)
        self.randomness_slider.sliderReleased.connect(self.on_slider_changed)
        lt.addWidget(QLabel("Randomness in Picking"), 0, 0, 1, 1)
        lt.addWidget(self.randomness_slider, 0, 1, 1, 1)

        self.simulations = QSlider( Qt.Horizontal, self)
        self.simulations.setRange(1, 20)
        self.simulations.sliderReleased.connect(self.on_slider_changed)
        lt.addWidget(QLabel("Number of Simulations"), 1,0, 1, 1)
        lt.addWidget(self.simulations, 1, 1, 1, 1)

        self.iteration_slider = QSlider( Qt.Horizontal, self)
        self.iteration_slider.setRange(0, 100)
        self.iteration_slider.valueChanged.connect(self.on_iteration_changed)
        lt.addWidget(QLabel("View Iteration"), 2,0, 1, 1)
        lt.addWidget(self.iteration_slider, 2, 1, 1, 1)

        self.n_nodes = QSlider(Qt.Horizontal, self)
        self.n_nodes.setRange(0, 1000)
        self.n_nodes.setValue(100)
        self.n_nodes.sliderReleased.connect(self.on_graph_changed)
        lt.addWidget(QLabel("Graph Nodes"), 3, 0, 1, 1)
        lt.addWidget(self.n_nodes, 3, 1, 1, 1)

        self.connectivity = QSlider(Qt.Horizontal, self)
        self.connectivity.setRange(1, 10)
        self.connectivity.setValue(1)
        self.connectivity.sliderReleased.connect(self.on_graph_changed)
        lt.addWidget(QLabel("Connectivity"), 4, 0, 1, 1)
        lt.addWidget(self.connectivity, 4, 1, 1, 1)

        entries = [
            "Barabasi Albert Graph",
            "Regular",
            "Erdös-Reny Graph"
        ]
        self.network_type = QComboBox(self)
        self.network_type.addItems(entries)
        self.network_type.currentTextChanged.connect(self.on_graph_changed)
        lt.addWidget(QLabel("Network Type"), 5, 0, 1, 1)
        lt.addWidget(self.network_type, 5, 1, 1, 1)

        self.setWidget(self.inner)

    def on_slider_changed(self):
        v = 0
        if self.randomness_slider.value() > 0:
            v = self.randomness_slider.value() / 100
        self.onRandomnessChanged.emit(v)

    def on_iteration_changed(self):
        self.onIterationChanged.emit(self.iteration_slider.value())

    def on_graph_changed(self):
        self.onGraphChanged.emit(self.n_nodes.value(), self.connectivity.value(), self.simulations.value(), self.network_type.currentText())


class MatplotlibWidget(QDockWidget):
    def __init__(self, parent):
        super(MatplotlibWidget, self).__init__(parent)
        self.setWindowTitle("Robustness")
        self.figure = plt.figure()
        self.figure.tight_layout()
        self.canvas = FigureCanvas(self.figure)
        self.ax1 = self.figure.add_subplot(221)
        self.ax2 = self.figure.add_subplot(222)
        self.ax3 = self.figure.add_subplot(212)
        self.setWidget(self.canvas)
        # self.plot(1,1)

    def plot(self, x1, y1):

        for ax in self.figure.axes:
            ax.clear()

        if isinstance(y1, dict):
            colmap = cm.get_cmap("Accent", len(y1.keys()))
            axes = [self.ax1, self.ax2, self.ax3]
            for i, (k, (xs, ys)) in enumerate(y1.items()):
                axes[i].set_title(k)
                axes[i].plot(xs, ys)#colmap(i), label=k
                # axes[i].legend()
                axes[i].set_xlim([0, np.amax(xs)])
        # else:
        #     self.ax1.plot(x1, y1, 'b.-')

        self.canvas.draw()



def my_exception_hook(exctype, value, traceback):
    # Print the error and traceback
    print((exctype, value, traceback))
    # Call the normal Exception hook after
    sys._excepthook(exctype, value, traceback)
    sys.exit(1)


if __name__ == '__main__':
    import sys

    sys._excepthook = sys.excepthook
    sys.excepthook = my_exception_hook

    app = QApplication(sys.argv)

    style_sheet = open("qt_stylesheet_very_dark.css", 'r')
    style_sheet = style_sheet.read()
    app.setStyleSheet(style_sheet)

    main = MainWindow(None)
    main.show()
    sys.exit(app.exec_())

"""
Modify base on:
http://stackoverflow.com/questions/36086361/embed-matplotlib-in-pyqt-with-multiple-plot/36093604

"""