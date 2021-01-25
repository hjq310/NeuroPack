from PyQt5 import QtGui, QtCore, QtWidgets
import sys
import os
import numpy as np
import json
import re
import imp
import pkgutil
import time
import pyqtgraph as pg

import arc1pyqt.Globals.fonts as fonts
import arc1pyqt.Globals.styles as styles
from arc1pyqt import state
HW = state.hardware
APP = state.app
CB = state.crossbar
from arc1pyqt import modutils
from arc1pyqt.modutils import BaseThreadWrapper, BaseProgPanel, \
        makeDeviceList, ModTag

THIS_DIR = os.path.dirname(__file__)
for ui in ['nnanalysis', 'nnvaravg', 'nnvaravgrow', 'nnvardiff',
    'nnvardiffrow', 'nnvarsnap', 'nnvarsnaprow']:

    modutils.compile_ui(os.path.join(THIS_DIR, 'uis', '%s.ui' % ui),
        os.path.join(THIS_DIR, '%s.py' % ui))

from .nnanalysis import Ui_NNAnalysis
from .nnvarsnap import Ui_NNVarSnap
from .nnvarsnaprow import Ui_NNVarSnapRow
from .nnvardiff import Ui_NNVarDiff
from .nnvardiffrow import Ui_NNVarDiffRow
from .nnvaravg import Ui_NNVarAvg
from .nnvaravgrow import Ui_NNVarAvgRow

from . import NeuroCores


def _log(*args, **kwargs):
    if bool(os.environ.get('NNDBG', False)):
        print(*args, file=sys.stderr, **kwargs)


class NetworkState(object):
    """
    NetworkState stores all the information for the state of the
    training process. All history is available.
    """

    def __init__(self, NETSIZE, DEPTH, epochs, labelCounter=1):
        super(NetworkState, self).__init__()
        self.weight_addresses = []
        self.weights = \
            np.array(NETSIZE*[NETSIZE*[epochs*labelCounter*[0.0]]])
        self.NeurAccum = np.zeros(shape=(epochs, NETSIZE))
        self.fireCells = np.array(epochs*labelCounter*[NETSIZE*[0.0]])
        self.firingCells = np.array(NETSIZE*[0.0])
        self.fireHist = np.array((DEPTH+1)*[NETSIZE*[0.0]])


class Network(BaseThreadWrapper):

    """
    This is the abstract represantation of the network. It includes all
    information about the training process, such as potentiation and
    depression potential, number of epochs, the connection matrix (mapping of
    neurons to devices), the size of the network (`NETSIZE`), as well as
    potentiation and depression windows.

    Network parameters are typically loaded from the network configuration file
    which is a JSON file including all the necessary arguments to initialise the
    network. The following arguments are always necessary, regardless whether they
    are used by the core or not

    LTPWIN and LTDWIN: potentiation and depression potential windows
    NETSIZE: network size (number of neurons)
    DEPTH: network depth (layers of neurons)

    Check `NeuroData/NeuroBase.json` for the absolute minimal configuration of a
    network. Arbitrary information can be included in the JSON file. For instance
    see `NeuroData/SevenMotif.json` for a configuration file with additional data.
    All JSON parameters are exposed to the core through the `Network.params` field.

    The connection matrix (`conn_mat`) maps neuros to devices and marks them as
    either excitatory or inhibitory. See `NeuroData/motif_connmat.txt` for an
    example.

    Argument `stimin` denotes forced stimulus of neurons. See file
    `NeuroData/motif_stim.txt` for an example of such file. Essentially it denotes
    at which timestamp certain neurons are forced to fire, regardless of their
    current state. Forced and induced stimulus is typically aggregated to find out
    the full stimulus status, although this is a process done by the network core.
    Evolution of the training process is guided almost fully by the network
    cores (which are placed under `NeuroCores`). This class does not do much
    apart from calling the core functions for each timestep and saving the data
    at the end of the process.

    It also exposes some convenience functions such as `read` and `pulse` that
    core implementers can call instead of fiddling with ArC1 internals.

    List of core-accessible values and functions

    * `Network.ConnMat`: The current connection matrix
    * `Network.stimin`: Forced stimulus as loaded from the stimulus file
    * `Network.epochs`: Total number of iterations
    * `Network.LTP_V` and `Network.LTD_V`: Potentiation and depression
      amplitude in Volts (these are set in the main neuropack UI)
    * `Network.LTP_pw` and `Network.LTD_pw`: Potentiation and depression
      pulse widths in seconds (again these are picked up from the main
      neuropack UI)
    * `Network.NETSIZE`: Size of the network
    * `Network.LTPWIN` and `Network.LTDWIN`: Potentiation and depression
       windows.
    * `Network.DEPTH`: Depth of the network
    * `Network.params`: This is a dict containing all JSON parameters as
      picked up from the configuration file.
    * `Network.state`: Complete history of weight and accumulators as well
      as calculated neuron firings. History state population is responsibility
      of the core.
    * `Network.pulse(w, b, A, pw)`: Function used to pulse device `(w,b)` with
      a voltage pulse of amplitude `A` and pulse width `pw` (volts and seconds).
      This will immediately return the new resistance of the device.
    * `Network.read(w, b)`: Read-out device `(w,b)` and get its resistance.

    """

    def __init__(self, conn_mat, stimin, data, params, tsteps, core, labelCounter=1):
        super(Network, self).__init__()

        self.ConnMat = conn_mat
        self.stimin = stimin

        self.LTP_V = data["LTP_V"]
        self.LTP_pw = data["LTP_pw"]
        self.LTD_V = data["LTD_V"]
        self.LTD_pw = data["LTD_pw"]
        self.epochs = data["epochs"]
        self.filename = data["fname"]

        # pop the core parameters into distinct fields
        self.NETSIZE = params.pop("NETSIZE")
        self.LTPWIN = params.pop("LTPWIN")
        self.LTDWIN = params.pop("LTDWIN")
        self.DEPTH = params.pop("DEPTH")

        # and bundle the rest under self.params
        self.params = params

        self.tsteps = tsteps
        self.rawin = np.array(self.NETSIZE*[0])
        self.Vread = HW.conf.Vread

        self.state = NetworkState(self.NETSIZE, self.DEPTH, \
            self.epochs, labelCounter)
        self.plot_counter_trigger = 100
        self.plot_counter = 0

        self.core = self.load_core(core)

    def log(self, *args, **kwargs):
        """ Write to stderr if CTSDBG is set"""

        _log(*args, **kwargs)

    def load_core(self, name):
        results = imp.find_module(name, NeuroCores.__path__)
        return imp.load_module(name, *results)

    @BaseThreadWrapper.runner
    def run(self):
        self.disableInterface.emit(True)

        self.log("Reading all devices and initialising weights")


        # For every neuron in the system.
        for postidx in range(len(self.ConnMat)):
            # For every presynaptic input the neuron receives.
            for preidx in np.where(self.ConnMat[:, postidx, 0] != 0)[0]:
                w, b=self.ConnMat[preidx, postidx, 0:2]
                self.read(w,b)
                self.state.weights[preidx, postidx, 0] = \
                    1.0/self.read(w, b)
                # store device address and neuron ids for easy access in
                # history_panel
                self.state.weight_addresses.append([[w,b],[preidx,postidx]])
        self.log("Done.")

        print("Starting Neural Net simulator")

        # Start Neural Net training

        self.core.init(self)

        for t in range(self.tsteps):
            self.rawin = self.state.firingCells
            self.log("---> Time step: %d RAWIN: %s STIMIN: %s" % (t, self.rawin, self.stimin[:, t]))
            self.core.neurons(self, t)
            self.core.plast(self, t)
            self.displayData.emit()

        self.log("Final reading of all devices")
        # For every neuron in the system.
        for postidx in range(len(self.ConnMat)):
            # For every presynaptic input the neuron receives.
            for preidx in np.where(self.ConnMat[:, postidx, 0] != 0)[0]:
                w,b=self.ConnMat[preidx, postidx, 0:2]
                self.read(w, b)

        # Save data if so requested
        if self.filename is not None:
            data = {}

            # metadata; this is a numpy structured array
            meta = np.array([(self.epochs, self.LTP_V, self.LTP_pw, self.LTD_V, self.LTD_pw)],
                    dtype=[('trials', 'u8'), ('LTP_V', 'f4'), ('LTP_pw', 'f4'),
                        ('LTD_V', 'f4'), ('LTD_pw', 'f4')])
            data['meta'] = meta

            # standard data first

            # all weights
            data['weights'] = self.state.weights
            # calculated stimuli for each step
            data['stimulus'] = self.stimin
            # history of cells that have fired
            data['fires'] = self.state.fireCells.T
            # accumulator snapshots
            data['accumulator'] = self.state.NeurAccum.T

            # and then any other arrays the core has produced
            additional_data = self.core.additional_data(self)

            if additional_data is not None:
                for (k, v) in self.core.additional_data(self):
                    data[k] = v

            np.savez_compressed(self.filename, **data)

        self.disableInterface.emit(False)
        self.finished.emit()

    def read(self, w, b):
        # read a device and return read value
        # update interface
        self.highlight.emit(w, b)

        Mnow = HW.ArC.read_one(w, b)

        self.sendData.emit(w, b, Mnow, self.Vread, 0, \
                'S R%d V=%.1f' % (HW.conf.readmode, HW.conf.Vread))
        self.updateTree.emit(w, b)

        return Mnow

    def pulse(self, w, b, A, pw):
        # apply a pulse and return
        # can instead apply any voltage series
        self.highlight.emit(w,b)

        Mnow = HW.ArC.pulseread_one(w, b, A, pw)

        self.sendData.emit(w, b, Mnow, A, pw, 'P')
        self.updateTree.emit(w, b)

        return Mnow


class NeuroPack(BaseProgPanel):

    def __init__(self, short=False):
        super().__init__(title="NeuroPack",
            description="Flexible neural nets", short=short)
        self.short = short
        self.base_conf_fname = None
        self.conn_matrix_fname = None
        self.stim_file_fname = None
        self.output_file_fname = None
        self.initUI()

        fname = os.path.join(THIS_DIR, "NeuroData", "NeuroBase.json")
        params = self.load_base_conf(os.path.join(THIS_DIR, "NeuroData",\
            "NeuroBase.json"))
        self.apply_base_conf(params, os.path.basename(fname), fname)

    def initUI(self):

        vbox1=QtWidgets.QVBoxLayout()

        titleLabel = QtWidgets.QLabel('NeuroPack')
        titleLabel.setFont(fonts.font1)
        descriptionLabel = QtWidgets.QLabel('Flexible neural net application module.')
        descriptionLabel.setFont(fonts.font3)
        descriptionLabel.setWordWrap(True)

        isInt=QtGui.QIntValidator()
        isFloat=QtGui.QDoubleValidator()

        gridLayout = QtWidgets.QGridLayout()
        gridLayout.setColumnStretch(0, 3)
        gridLayout.setColumnStretch(1, 1)
        gridLayout.setColumnStretch(2, 1)
        gridLayout.setColumnStretch(3, 1)

        #setup a line separator
        lineLeft=QtWidgets.QFrame()
        lineLeft.setFrameShape(QtWidgets.QFrame.VLine)
        lineLeft.setFrameShadow(QtWidgets.QFrame.Raised)
        lineLeft.setLineWidth(1)
        gridLayout.addWidget(lineLeft, 0, 2, 2, 1)

        ################################################ LOAD ##############
        self.push_load_base_conf = QtWidgets.QPushButton("Load Base conf.")
        self.push_load_base_conf.clicked.connect(self.open_base_conf)
        self.base_conf_filename = QtWidgets.QLabel("Filename")
        gridLayout.addWidget(self.push_load_base_conf, 0, 1)
        gridLayout.addWidget(self.base_conf_filename, 0, 0)

        self.push_load_conn_matrix = QtWidgets.QPushButton("Load Conn. Matrix")
        self.push_load_conn_matrix.clicked.connect(self.open_conn_matrix)
        self.matrix_filename = QtWidgets.QLabel("Filename")
        gridLayout.addWidget(self.push_load_conn_matrix, 1, 1)
        gridLayout.addWidget(self.matrix_filename, 1, 0)

        self.push_load_stim_file = QtWidgets.QPushButton("Load Stim. File")
        self.push_load_stim_file.clicked.connect(self.open_stim_file)
        self.stim_filename = QtWidgets.QLabel("Filename")
        gridLayout.addWidget(self.push_load_stim_file, 2 ,1)
        gridLayout.addWidget(self.stim_filename, 2, 0)

        self.check_save_data = QtWidgets.QCheckBox("Save to:")
        self.check_save_data.clicked.connect(self.check_save_data_clicked)
        self.push_save_filename = QtWidgets.QPushButton("No file selected")
        self.push_save_filename.clicked.connect(self.load_output_file)
        self.push_save_filename.setEnabled(False)
        gridLayout.addWidget(self.check_save_data, 4, 0)
        gridLayout.addWidget(self.push_save_filename, 4, 1)

        self.push_show_analysis_tool = QtWidgets.QPushButton("Start analysis tool")
        self.push_show_analysis_tool.clicked.connect(self.startAnalysisTool)
        gridLayout.addWidget(self.push_show_analysis_tool, 5, 0, 1, 2)

        ####################################################################

        ################################################## CORES ###########

        self.rulesCombo = QtWidgets.QComboBox()
        for _, name, is_pkg in pkgutil.iter_modules(NeuroCores.__path__):
            if not is_pkg and name.startswith("core_"):
                self.rulesCombo.addItem(name.replace("core_", ""), name)

        gridLayout.addWidget(QtWidgets.QLabel("Network core:"), 3, 0)
        gridLayout.addWidget(self.rulesCombo, 3, 1)

        ####################################################################

        leftLabels=[]
        self.leftEdits=[]

        rightLabels=['LTP pulse voltage (V)', \
                    'LTP pulse width (us)',\
                    'LTD pulse voltage (V)',\
                    'LTD pulse width (us)', \
                    'Trials']

        self.rightEdits=[]

        leftInit=  []
        rightInit= ['1.1', \
                    '10',\
                    '-1.1',\
                    '10',\
                    '100',\
                    '10']

        #setup a line separator
        lineLeft = QtWidgets.QFrame()
        lineLeft.setFrameShape(QtWidgets.QFrame.VLine);
        lineLeft.setFrameShadow(QtWidgets.QFrame.Raised);
        lineLeft.setLineWidth(1)

        gridLayout.addWidget(lineLeft, 0, 2, 7, 1)

        for i in range(len(leftLabels)):
            lineLabel=QtWidgets.QLabel()
            lineLabel.setText(leftLabels[i])
            gridLayout.addWidget(lineLabel, i,0)

            lineEdit=QtWidgets.QLineEdit()
            lineEdit.setText(leftInit[i])
            lineEdit.setValidator(isFloat)
            self.leftEdits.append(lineEdit)
            gridLayout.addWidget(lineEdit, i,1)

        for i in range(len(rightLabels)):
            lineLabel=QtWidgets.QLabel()
            lineLabel.setText(rightLabels[i])
            gridLayout.addWidget(lineLabel, i,4)

            lineEdit=QtWidgets.QLineEdit()
            lineEdit.setText(rightInit[i])
            lineEdit.setValidator(isFloat)
            self.rightEdits.append(lineEdit)
            gridLayout.addWidget(lineEdit, i,5)

        self.LTP_V=float(self.rightEdits[0].text())
        self.LTP_pw=float(self.rightEdits[1].text())/1000000
        self.LTD_V=float(self.rightEdits[2].text())
        self.LTD_pw=float(self.rightEdits[3].text())/1000000

        ################################################ LTD/LTP ###########

        vbox1.addWidget(titleLabel)
        vbox1.addWidget(descriptionLabel)

        self.vW=QtWidgets.QWidget()
        self.vW.setLayout(gridLayout)
        self.vW.setContentsMargins(0,0,0,0)

        scrlArea=QtWidgets.QScrollArea()
        scrlArea.setWidget(self.vW)
        scrlArea.setContentsMargins(0,0,0,0)
        scrlArea.setWidgetResizable(False)
        scrlArea.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

        scrlArea.installEventFilter(self)

        vbox1.addWidget(scrlArea)
        vbox1.addStretch()

        if not self.short:
            self.hboxProg = QtWidgets.QHBoxLayout()

            push_train = QtWidgets.QPushButton('Train Network')
            push_train.setStyleSheet(styles.btnStyle)
            push_train.clicked.connect(self.runTrain)
            self.hboxProg.addWidget(push_train)

            vbox1.addLayout(self.hboxProg)

        self.labelCounter=1

        self.setLayout(vbox1)
        self.gridLayout=gridLayout

    # def update_learning_rule(self):
    #     pass

    def gather_data(self):

        if self.check_save_data.isChecked():
            fname = self.output_file_fname
        else:
            fname = None

        return { \
            "LTP_V": float(self.rightEdits[0].text()), \
            "LTP_pw": float(self.rightEdits[1].text())/1000000, \
            "LTD_V": float(self.rightEdits[2].text()),\
            "LTD_pw": float(self.rightEdits[3].text())/1000000,\
            "epochs": int(self.rightEdits[4].text()),
            "fname": fname
        }

    def gather_params(self):
        return json.load(open(self.base_conf_fname))

    def runTrain(self):

        def _check_output_file(fname):

            if fname is None:
                return False

            if os.path.exists(fname) and os.stat(fname).st_size > 0:
                btns = QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
                text = "File exists and is of non-zero size. Overwrite?"
                reply = QtWidgets.QMessageBox.question(self, "File exists", \
                        text, btns)
                if reply == QtWidgets.QMessageBox.Yes:
                    return True
                return False
            else:
                return True

        if (self.conn_matrix_fname is None) or (self.stim_file_fname is None):
            errMessage = QtWidgets.QMessageBox()
            errMessage.setText("No connection matrix or stimulus file")
            errMessage.setIcon(QtWidgets.QMessageBox.Critical)
            errMessage.setWindowTitle("Error")
            errMessage.exec_()
            return

        data = self.gather_data()
        params = self.gather_params()

        if not _check_output_file(data['fname']):
            data['fname'] = None

        # epochs times the nr of time steps
        # that are defined in the stimulus file
        tsteps = data["epochs"] * self.labelCounter

        # Reload the stimulus file to account for any changes in the epochs
        # Could possibly check if the field is "tainted" before loading to
        # avoid accessing the file again
        self.stimin = self.load_stim_file(self.stim_file_fname, \
            params["NETSIZE"], data["epochs"])

        if HW.ArC is not None:
            coreIdx = self.rulesCombo.currentIndex()
            coreName = self.rulesCombo.itemData(coreIdx)

            network = Network(self.ConnMat, self.stimin, data, params, \
                tsteps, coreName, self.labelCounter)
            self.execute(network, network.run, True)

    def load_base_conf(self, fname):
        return json.load(open(fname))

    def open_base_conf(self):
        path = QtCore.QFileInfo(QtWidgets.QFileDialog().getOpenFileName(self,\
            'Open base configuration', THIS_DIR, filter="*.json")[0])
        name = path.fileName()

        try:
            data = self.load_base_conf(path.filePath())
            for x in ["LTDWIN", "LTPWIN", "NETSIZE", "DEPTH"]:
                if x not in data.keys():
                    errMessage = QtWidgets.QMessageBox()
                    errMessage.setText("Missing required parameter %s" % x)
                    errMessage.setIcon(QtWidgets.QMessageBox.Critical)
                    errMessage.setWindowTitle("Error")
                    errMessage.exec_()
                    return
            self.apply_base_conf(data, name, path.filePath())
        except Exception as exc:
            _log("!!", exc)

    def apply_base_conf(self, params, name, path):
        self.base_conf_fname = str(path)
        self.base_conf_filename.setText(name)
        if self.conn_matrix_fname is not None:
            data = self.gather_data()
            res = self.load_conn_matrix(self.conn_matrix_fname,\
                params["NETSIZE"])
            self.apply_conn_matrix(res)

        if self.stim_file_fname is not None:
            data = self.gather_data()
            stims = self.load_stim_file(self.stim_file_fname, params["NETSIZE"],\
                data["epochs"])
            self.apply_stim_file(stims)

    def load_stim_file(self, fname, NETSIZE, epochs):
        _log("Allocating stimin")
        stimin = np.array(NETSIZE*[epochs*[0]])

        with open(fname, 'r') as f:
            _log("File opened")
            for line in f:
                line = line.strip()
                if (line[0] != "\n") and (line[0] != "#"):
                    # split into timestamp - list of neurons IDs scheduled to spike
                    timestamp, neuronIDs = re.split("\s*-\s*", line)
                    timestamp = int(timestamp)
                    if timestamp >= epochs:
                        break
                    _log(timestamp, neuronIDs)
                    # split the string into an int list of neurons
                    spikeNeuronID = [int(x) - 1 for x in re.split("\s*,\s*", neuronIDs.strip())]
                    for i, spiker in enumerate(spikeNeuronID):
                        stimin[spiker, timestamp] = 1
        return stimin

    def open_stim_file(self):
        _log("Loading stimulation file...")

        params = self.gather_params()
        data = self.gather_data()

        path = QtCore.QFileInfo(QtWidgets.QFileDialog().getOpenFileName(self,\
            'Open stimulation file', THIS_DIR, filter="*.txt")[0])
        name = path.fileName()

        error = False
        try:
            res = self.load_stim_file(path.filePath(), params["NETSIZE"], \
                data["epochs"])
        except Exception as exc:
            _log(exc)
            error = True

        #print(self.stimin)

        if error:
            self.stimin = np.array(params["NETSIZE"]*[data["epochs"]*[0]])
            errMessage = QtWidgets.QMessageBox()
            errMessage.setText("Invalid network stimulation file! Possible problem with syntax.")
            errMessage.setIcon(QtWidgets.QMessageBox.Critical)
            errMessage.setWindowTitle("Error")
            errMessage.exec_()
        else:
            self.apply_stim_file(res, name, path.filePath())

        _log("done")

    def apply_stim_file(self, stim, name=None, path=None):
        self.stimin = stim
        if name is not None:
            self.stim_filename.setText(name)
        if path is not None:
            self.stim_file_fname = str(path)

    def load_conn_matrix(self, fname, NETSIZE):
        ConnMat = np.array(NETSIZE*[NETSIZE*[3*[0]]])

        with open(fname, 'r') as f:
            for line in f:
                if (line[0] != "\n") and (line[0] != "#"):
                    preid, postid, w, b, type_of=line.split(", ")
                    ConnMat[int(preid) - 1, int(postid) - 1] = \
                        [int(w), int(b), int(type_of)]

        return ConnMat

    def open_conn_matrix(self):
        _log("Loading connectivity matrix...")

        params = self.gather_params()

        # self.ConnMat=np.array(params["NETSIZE"]*[params["NETSIZE"]*[3*[0]]])

        path = QtCore.QFileInfo(QtWidgets.QFileDialog().getOpenFileName(self,\
            'Open connectivity matrix file', THIS_DIR, filter="*.txt")[0])
        name=path.fileName()

        error = False

        try:
            res = self.load_conn_matrix(path.filePath(), params["NETSIZE"])
        except Exception as exc:
            error = True
            _log(exc)

        if error:
            self.ConnMat=np.array(params["NETSIZE"]*[params["NETSIZE"]*[3*[0]]])
            errMessage = QtWidgets.QMessageBox()
            errMessage.setText("Invalid connectivity matrix file! Possible problem with syntax.")
            errMessage.setIcon(QtWidgets.QMessageBox.Critical)
            errMessage.setWindowTitle("Error")
            errMessage.exec_()
        else:
            # self.matrix_filename.setText(name)
            # self.ConnMat = res
            # self.conn_matrix_fname = path.filePath()
            self.apply_conn_matrix(res, name, path.filePath())

        _log("done")

    def apply_conn_matrix(self, matrix, name=None, path=None):
        self.ConnMat = matrix
        if path is not None:
            self.matrix_filename.setText(name)
        if name is not None:
            self.conn_matrix_fname = str(path)

    def check_save_data_clicked(self, checked):
        self.push_save_filename.setEnabled(checked)

    def load_output_file(self):

        if self.output_file_fname is not None:
            curpath = os.path.dirname(self.output_file_fname)
        else:
            curpath = ''

        path = QtCore.QFileInfo(QtWidgets.QFileDialog().getSaveFileName(self,\
            'Save to...', curpath, "Numpy arrays (*.npz)")[0])
        fname = path.fileName()

        if fname is None or len(fname) == 0:
            if self.output_file_fname is None:
                self.check_save_data.setChecked(False)
                self.push_save_filename.setEnabled(False)
            return

        self.output_file_fname = path.filePath()
        self.push_save_filename.setText(fname)

        _log("Set output file to %s..." % path.fileName())

    def startAnalysisTool(self):
        self._analysisWindow = QtWidgets.QMainWindow()
        self._analysisWindow.setWindowTitle("NeuroPack Analysis tool")
        self._analysisWindow.setCentralWidget(NeuroAnalysis())
        self._analysisWindow.show()

    def eventFilter(self, object, event):
        if event.type()==QtCore.QEvent.Resize:
            self.vW.setFixedWidth(event.size().width() - \
                object.verticalScrollBar().width())
        return False

    def disableProgPanel(self,state):
        self.hboxProg.setEnabled(not state)


class NeuroVarSnapRowWidget(Ui_NNVarSnapRow, QtWidgets.QWidget):

    def __init__(self, dataset, parent=None):
        super(NeuroVarSnapRowWidget, self).__init__(parent=parent)

        self.dataset = None
        self.selected = False
        self.setupUi(self)
        self.plotWidget = None

        self.updateDataset(dataset)
        self.setSelected(self.selected)
        self.stepSlider.valueChanged.connect(self.sliderChanged)
        self.stepSpinBox.valueChanged.connect(self.stepSpinBoxChanged)
        self.variableSelectionCombo.currentIndexChanged.connect(self.variableChanged)

    def _clearGraphs(self):
        for idx in reversed(range(self.graphHolderLayout.count())):
            wdg = self.graphHolderLayout.itemAt(idx).widget()
            self.graphHolderLayout.removeWidget(wdg)
            wdg.setParent(None)

    def stepSpinBoxChanged(self, val):
        self.stepSlider.blockSignals(True)
        self.stepSlider.setValue(val)
        self.stepSlider.blockSignals(False)
        self.updatePlotToStep(val)

    def sliderChanged(self, val):
        self.stepSpinBox.blockSignals(True)
        self.stepSpinBox.setValue(val)
        self.stepSpinBox.blockSignals(False)
        self.updatePlotToStep(val)

    def setStep(self, step):
        self.stepSlider.setValue(step)
        self.stepSpinBox.setValue(step)
        self.updatePlotToStep(step)

    def _updateGraph(self, data, step):

        plotArgs = {'pen': None, 'symbolPen': None, 'symbolBrush': (255,0,0), \
                'symbol':'+'}

        # determine the plot type for the existing graph, if any
        if self.plotWidget is None:
            wdgDim = -1 # no widget yet
        else:
            if isinstance(self.plotWidget, pg.PlotWidget):
                wdgDim = 2
            else:
                wdgDim = 3

        # and either generate a new plot or update the existing one
        # (if dimensions match)
        if wdgDim != len(data.shape):
            # changed from 2 to 3D or vice-versa; need to update widget
            self._clearGraphs()
            if len(data.shape) == 2:
                wdg = pg.PlotWidget()
                wdg.plot(np.arange(len(data.T[step]))+1, data.T[step], **plotArgs)
            elif len(data.shape) == 3:
                wdg = pg.ImageView()
                wdg.ui.menuBtn.hide()
                wdg.ui.roiBtn.hide()
                wdg.setImage(data.T[step])
            else:
                # more dimensions unable to visualise
                wdg = QtWidgets.QLabel("Cannot visualise > 3 dimensions")
            self.graphHolderLayout.addWidget(wdg)
            self.plotWidget = wdg
        else:
            if wdgDim == 2:
                self.plotWidget.plot(np.arange(len(data.T[step]))+1,
                        data.T[step], clear=True, **plotArgs)
            elif wdgDim == 3:
                self.plotWidget.setImage(data.T[step])

    def updatePlotToStep(self, step):
        idx = self.variableSelectionCombo.currentIndex()
        data = self.variableSelectionCombo.itemData(idx)

        self._updateGraph(data, step)

    def updateDataset(self, dataset):

        if dataset is None:
            return

        self.variableSelectionCombo.clear()

        for k in dataset.files:
            if k == 'meta':
                continue
            self.variableSelectionCombo.addItem(k, dataset[k])

        self.stepSlider.setMinimum(0)
        self.stepSlider.setMaximum(int(dataset['meta']['trials'][0])-1)

        self.stepSpinBox.setMinimum(0)
        self.stepSpinBox.setMaximum(int(dataset['meta']['trials'][0])-1)

        self.dataset = dataset

    def variableChanged(self, idx):
        currentStep = self.stepSpinBox.value()
        self.updatePlotToStep(currentStep)

    def setSelected(self, status):
        self.selected = status
        if self.selected:
            colour = "#F00"
        else:
            colour = "#000"

        self.rowFrame.setStyleSheet("#rowFrame {border: 1px solid %s}" % colour)


class NeuroVarSnapWidget(Ui_NNVarSnap, QtWidgets.QWidget):

    def __init__(self, dataset, parent=None):
        super(NeuroVarSnapWidget, self).__init__(parent=parent)
        self.dataset = dataset

        self.rows =  []
        self.selectedRow = None

        self.setupUi(self)

        self.lockStepsCheckBox.stateChanged.connect(self.lockStepsChecked)
        self.globalStepSlider.valueChanged.connect(self.stepSliderChanged)
        self.globalStepSpinBox.valueChanged.connect(self.stepSpinBoxChanged)
        self.addRowButton.clicked.connect(self.addRow)
        self.deleteRowButton.clicked.connect(self.removeRow)

    def mousePressEvent(self, evt):
        for (idx, row) in enumerate(self.rows):
            if row.underMouse():
                self.selectedRow = idx
            row.setSelected(row.underMouse())

    def lockStepsChecked(self):
        checked = self.lockStepsCheckBox.isChecked()
        self.globalStepSlider.setEnabled(checked)
        self.globalStepSpinBox.setEnabled(checked)

    def stepSliderChanged(self, val):
        self.globalStepSpinBox.blockSignals(True)
        self.globalStepSpinBox.setValue(val)
        self.globalStepSpinBox.blockSignals(False)
        self.updateGlobalStep(val)

    def stepSpinBoxChanged(self, val):
        self.globalStepSlider.blockSignals(True)
        self.globalStepSlider.setValue(val)
        self.globalStepSlider.blockSignals(False)
        self.updateGlobalStep(val)

    def updateGlobalStep(self, step):
        for row in self.rows:
            row.setStep(step)

    def updateDataset(self, dataset):
        if dataset is None:
            return

        for row in self.rows:
            row.updateDataset(dataset)

        self.globalStepSlider.setMinimum(0)
        self.globalStepSlider.setMaximum(int(dataset['meta']['trials'][0])-1)
        self.globalStepSpinBox.setMinimum(0)
        self.globalStepSpinBox.setMaximum(int(dataset['meta']['trials'][0])-1)
        self.dataset = dataset

    def addRow(self):
        self.rows.append(NeuroVarSnapRowWidget(self.dataset))
        self.mainSnapLayout.addWidget(self.rows[-1])
        self.rows[-1].setMinimumHeight(350)

    def removeRow(self):
        if self.selectedRow is None:
            return

        wdg = self.rows.pop(self.selectedRow)
        self.mainSnapLayout.removeWidget(wdg)
        wdg.setParent(None)
        self.selectedRow = None
        for row in self.rows:
            row.setSelected(False)


class NeuroVarDiffRowWidget(Ui_NNVarDiffRow, QtWidgets.QWidget):

    def __init__(self, dataset, parent=None):
        super(NeuroVarDiffRowWidget, self).__init__(parent=parent)
        self.dataset = None
        self.selected = False
        self.setupUi(self)
        self.plotWidget = None
        self.updateDataset(dataset)
        self.setSelected(self.selected)

        self.initialStepSpinBox.valueChanged.connect(\
                lambda val: self.stepSpinBoxChanged(self.initialStepSpinBox, self.initialStepSlider, val))
        self.initialStepSlider.valueChanged.connect(\
                lambda val: self.stepSliderChanged(self.initialStepSpinBox, self.initialStepSlider, val))
        self.finalStepSpinBox.valueChanged.connect(\
                lambda val: self.stepSpinBoxChanged(self.finalStepSpinBox, self.finalStepSlider, val))
        self.finalStepSlider.valueChanged.connect(\
                lambda val: self.stepSliderChanged(self.finalStepSpinBox, self.finalStepSlider, val))
        self.variableSelectionCombo.currentIndexChanged.connect(self.variableChanged)

    def _clearGraphs(self):
        for idx in reversed(range(self.graphHolderLayout.count())):
            wdg = self.graphHolderLayout.itemAt(idx).widget()
            self.graphHolderLayout.removeWidget(wdg)
            wdg.setParent(None)

    def updateDataset(self, dataset):

        if dataset is None:
            return

        self.variableSelectionCombo.clear()

        for k in dataset.files:
            if k == 'meta':
                continue
            self.variableSelectionCombo.addItem(k, dataset[k])

        self.initialStepSlider.setMinimum(0)
        self.finalStepSlider.setMinimum(0)
        self.initialStepSlider.setMaximum(int(dataset['meta']['trials'][0])-1)
        self.finalStepSlider.setMaximum(int(dataset['meta']['trials'][0])-1)

        self.initialStepSpinBox.setMinimum(0)
        self.finalStepSpinBox.setMinimum(0)
        self.initialStepSpinBox.setMaximum(int(dataset['meta']['trials'][0])-1)
        self.finalStepSpinBox.setMaximum(int(dataset['meta']['trials'][0])-1)

    def setSelected(self, status):
        self.selected = status
        if self.selected:
            colour = "#F00"
        else:
            colour = "#000"

        self.rowFrame.setStyleSheet("#rowFrame {border: 1px solid %s}" % colour)

    def variableChanged(self, idx):
        initialStep = self.initialStepSpinBox.value()
        finalStep = self.finalStepSpinBox.value()
        self.updatePlotToStep(initialStep, finalStep)

    def stepSpinBoxChanged(self, box, slider, val):
        slider.blockSignals(True)
        slider.setValue(val)
        slider.blockSignals(False)

        initialStep = self.initialStepSpinBox.value()
        finalStep = self.finalStepSpinBox.value()
        self.updatePlotToStep(initialStep, finalStep)

    def stepSliderChanged(self, box, slider, val):
        box.blockSignals(True)
        box.setValue(val)
        box.blockSignals(False)

        initialStep = self.initialStepSpinBox.value()
        finalStep = self.finalStepSpinBox.value()
        self.updatePlotToStep(initialStep, finalStep)

    def updatePlotToStep(self, initialStep, finalStep):
        idx = self.variableSelectionCombo.currentIndex()
        data = self.variableSelectionCombo.itemData(idx)

        self._updateGraph(data, initialStep, finalStep)

    def _updateGraph(self, data, initialStep, finalStep):

        plotArgs = {'pen': None, 'symbolPen': None, 'symbolBrush': (255,0,0), \
                'symbol':'+'}

        # determine the plot type for the existing graph, if any
        if self.plotWidget is None:
            wdgDim = -1 # no widget yet
        else:
            if isinstance(self.plotWidget, pg.PlotWidget):
                wdgDim = 2
            else:
                wdgDim = 3

        # and either generate a new plot or update the existing one
        # (if dimensions match)
        if wdgDim != len(data.shape):
            # changed from 2 to 3D or vice-versa; need to update widget
            self._clearGraphs()
            if len(data.shape) == 2:
                wdg = pg.PlotWidget()
                diff = data.T[finalStep] - data.T[initialStep]
                # regardless of where you are in time the length of
                # the data will always be the same; so either finalStep
                # or initialStep is good enough for the X-axis
                wdg.plot(np.arange(len(data.T[finalStep]))+1, diff, **plotArgs)
            elif len(data.shape) == 3:
                wdg = pg.ImageView()
                wdg.ui.menuBtn.hide()
                wdg.ui.roiBtn.hide()
                diff = data.T[finalStep] - data.T[initialStep]
                wdg.setImage(diff)
            else:
                # more dimensions unable to visualise
                wdg = QtWidgets.QLabel("Cannot visualise > 3 dimensions")
            self.graphHolderLayout.addWidget(wdg)
            self.plotWidget = wdg
        else:
            if wdgDim == 2:
                diff = data.T[finalStep] - data.T[initialStep]
                # regardless of where you are in time the length of
                # the data will always be the same; so either finalStep
                # or initialStep is good enough for the X-axis
                self.plotWidget.plot(np.arange(len(data.T[finalStep]))+1,
                        diff, clear=True, **plotArgs)
            elif wdgDim == 3:
                diff = data.T[finalStep] - data.T[initialStep]
                self.plotWidget.setImage(diff)


class NeuroVarDiffWidget(Ui_NNVarDiff, QtWidgets.QWidget):

    def __init__(self, dataset, parent=None):
        super(NeuroVarDiffWidget, self).__init__(parent=parent)
        self.dataset = dataset

        self.rows =  []
        self.selectedRow = None

        self.setupUi(self)

        self.addRowButton.clicked.connect(self.addRow)
        self.deleteRowButton.clicked.connect(self.removeRow)

    def mousePressEvent(self, evt):
        for (idx, row) in enumerate(self.rows):
            if row.underMouse():
                self.selectedRow = idx
            row.setSelected(row.underMouse())

    def updateDataset(self, dataset):
        if dataset is None:
            return

        for row in self.rows:
            row.updateDataset(dataset)

        self.dataset = dataset

    def addRow(self):
        self.rows.append(NeuroVarDiffRowWidget(self.dataset))
        self.mainSnapLayout.addWidget(self.rows[-1])
        self.rows[-1].setMinimumHeight(350)

    def removeRow(self):
        if self.selectedRow is None:
            return

        wdg = self.rows.pop(self.selectedRow)
        self.mainSnapLayout.removeWidget(wdg)
        wdg.setParent(None)
        self.selectedRow = None
        for row in self.rows:
            row.setSelected(False)


class NeuroVarAvgRowWidget(Ui_NNVarAvgRow, QtWidgets.QWidget):

    def __init__(self, dataset, parent=None):
        super(NeuroVarAvgRowWidget, self).__init__(parent=parent)
        self.dataset = None
        self.selected = False
        self.setupUi(self)
        self.plotWidget = None
        self.updateDataset(dataset)
        self.setSelected(self.selected)

        self.fromStepSpinBox.valueChanged.connect(\
                lambda val: self.stepSpinBoxChanged(self.fromStepSpinBox, self.fromStepSlider, val))
        self.fromStepSlider.valueChanged.connect(\
                lambda val: self.stepSliderChanged(self.fromStepSpinBox, self.fromStepSlider, val))
        self.toStepSpinBox.valueChanged.connect(\
                lambda val: self.stepSpinBoxChanged(self.toStepSpinBox, self.toStepSlider, val))
        self.toStepSlider.valueChanged.connect(\
                lambda val: self.stepSliderChanged(self.toStepSpinBox, self.toStepSlider, val))
        self.variableSelectionCombo.currentIndexChanged.connect(self.variableChanged)

    def _clearGraphs(self):
        for idx in reversed(range(self.graphHolderLayout.count())):
            wdg = self.graphHolderLayout.itemAt(idx).widget()
            self.graphHolderLayout.removeWidget(wdg)
            wdg.setParent(None)

    def updateDataset(self, dataset):

        if dataset is None:
            return

        self.variableSelectionCombo.clear()

        for k in dataset.files:
            if k == 'meta':
                continue
            self.variableSelectionCombo.addItem(k, dataset[k])

        self.fromStepSlider.setMinimum(0)
        self.toStepSlider.setMinimum(0)
        self.fromStepSlider.setMaximum(int(dataset['meta']['trials'][0])-1)
        self.toStepSlider.setMaximum(int(dataset['meta']['trials'][0])-1)

        self.fromStepSpinBox.setMinimum(0)
        self.toStepSpinBox.setMinimum(0)
        self.fromStepSpinBox.setMaximum(int(dataset['meta']['trials'][0])-1)
        self.toStepSpinBox.setMaximum(int(dataset['meta']['trials'][0])-1)

    def setSelected(self, status):
        self.selected = status
        if self.selected:
            colour = "#F00"
        else:
            colour = "#000"

        self.rowFrame.setStyleSheet("#rowFrame {border: 1px solid %s}" % colour)

    def variableChanged(self, idx):
        fromStep = self.fromStepSpinBox.value()
        toStep = self.toStepSpinBox.value()
        self.updatePlotToStep(fromStep, toStep)

    def stepSpinBoxChanged(self, box, slider, val):
        slider.blockSignals(True)
        slider.setValue(val)
        slider.blockSignals(False)

        fromStep = self.fromStepSpinBox.value()
        toStep = self.toStepSpinBox.value()
        self.updatePlotToStep(fromStep, toStep)

    def stepSliderChanged(self, box, slider, val):
        box.blockSignals(True)
        box.setValue(val)
        box.blockSignals(False)

        fromStep = self.fromStepSpinBox.value()
        toStep = self.toStepSpinBox.value()
        self.updatePlotToStep(fromStep, toStep)

    def setSteps(self, fromStep, toStep):
        self.fromStepSlider.blockSignals(True)
        self.toStepSlider.blockSignals(True)
        self.fromStepSpinBox.blockSignals(True)
        self.toStepSpinBox.blockSignals(True)

        self.fromStepSlider.setValue(fromStep)
        self.fromStepSpinBox.setValue(fromStep)
        self.toStepSlider.setValue(toStep)
        self.toStepSpinBox.setValue(toStep)
        self.updatePlotToStep(fromStep, toStep)

        self.fromStepSlider.blockSignals(False)
        self.toStepSlider.blockSignals(False)
        self.fromStepSpinBox.blockSignals(False)
        self.toStepSpinBox.blockSignals(False)

    def updatePlotToStep(self, fromStep, toStep):
        idx = self.variableSelectionCombo.currentIndex()
        data = self.variableSelectionCombo.itemData(idx)

        self._updateGraph(data, fromStep, toStep)

    def _updateGraph(self, data, fromStep, toStep):

        if toStep < fromStep:
            # swap variables if from > to
            toStep, fromStep = fromStep, toStep

        plotArgs = {'pen': None, 'symbolPen': None, 'symbolBrush': (255,0,0), \
                'symbol':'+'}

        # determine the plot type for the existing graph, if any
        if self.plotWidget is None:
            wdgDim = -1 # no widget yet
        else:
            if isinstance(self.plotWidget, pg.PlotWidget):
                wdgDim = 2
            else:
                wdgDim = 3

        # and either generate a new plot or update the existing one
        # (if dimensions match)
        if wdgDim != len(data.shape):
            # changed from 2 to 3D or vice-versa; need to update widget
            self._clearGraphs()
            if len(data.shape) == 2:
                wdg = pg.PlotWidget()
                # nothing to average between identical timesteps
                if fromStep == toStep:
                    avg = data[:,fromStep]
                else:
                    avg = np.average(data[:, fromStep:toStep], axis=1)
                # regardless of where you are in time the length of
                # the data will always be the same; so either finalStep
                # or initialStep is good enough for the X-axis
                wdg.plot(np.arange(len(data.T[fromStep]))+1, avg, **plotArgs)
            elif len(data.shape) == 3:
                wdg = pg.ImageView()
                wdg.ui.menuBtn.hide()
                wdg.ui.roiBtn.hide()
                if fromStep == toStep:
                    avg = data[:,:,fromStep]
                else:
                    avg = np.average(data[:,:,fromStep:toStep], axis=2)
                wdg.setImage(avg)
            else:
                # more dimensions unable to visualise
                wdg = QtWidgets.QLabel("Cannot visualise > 3 dimensions")
            self.graphHolderLayout.addWidget(wdg)
            self.plotWidget = wdg
        else:
            if wdgDim == 2:
                if fromStep == toStep:
                    avg = data[:,fromStep]
                else:
                    avg = np.average(data[:, fromStep:toStep], axis=1)
                # regardless of where you are in time the length of
                # the data will always be the same; so either finalStep
                # or initialStep is good enough for the X-axis
                self.plotWidget.plot(np.arange(len(data.T[fromStep]))+1,
                        avg, clear=True, **plotArgs)
            elif wdgDim == 3:
                if fromStep == toStep:
                    avg = data[:,:,fromStep]
                else:
                    avg = np.average(data[:,:,fromStep:toStep], axis=2)
                self.plotWidget.setImage(avg)


class NeuroVarAvgWidget(Ui_NNVarAvg, QtWidgets.QWidget):

    def __init__(self, dataset, parent=None):
        super(NeuroVarAvgWidget, self).__init__(parent=parent)
        self.dataset = dataset

        self.rows =  []
        self.selectedRow = None

        self.setupUi(self)

        self.lockStepsCheckBox.stateChanged.connect(self.lockStepsChecked)
        self.globalFromSpinBox.valueChanged.connect(self.globalFromSpinBoxChanged)
        self.globalToSpinBox.valueChanged.connect(self.globalToSpinBoxChanged)
        self.globalFromSlider.valueChanged.connect(self.globalFromSliderChanged)
        self.globalToSlider.valueChanged.connect(self.globalToSliderChanged)
        self.addRowButton.clicked.connect(self.addRow)
        self.deleteRowButton.clicked.connect(self.removeRow)

    def mousePressEvent(self, evt):
        for (idx, row) in enumerate(self.rows):
            if row.underMouse():
                self.selectedRow = idx
            row.setSelected(row.underMouse())

    def lockStepsChecked(self):
        checked = self.lockStepsCheckBox.isChecked()
        self.globalFromSlider.setEnabled(checked)
        self.globalToSlider.setEnabled(checked)
        self.globalFromSpinBox.setEnabled(checked)
        self.globalToSpinBox.setEnabled(checked)

    def globalFromSliderChanged(self, val):
        toStep = self.globalToSpinBox.value()

        self.globalFromSpinBox.blockSignals(True)
        self.globalFromSpinBox.setValue(val)
        self.globalFromSpinBox.blockSignals(False)
        self.updateGlobalSteps(val, toStep)

    def globalToSliderChanged(self, val):
        fromStep = self.globalFromSpinBox.value()

        self.globalToSpinBox.blockSignals(True)
        self.globalToSpinBox.setValue(val)
        self.globalToSpinBox.blockSignals(False)
        self.updateGlobalSteps(fromStep, val)

    def globalFromSpinBoxChanged(self, val):
        toStep = self.globalToSpinBox.value()

        self.globalFromSlider.blockSignals(True)
        self.globalFromSlider.setValue(val)
        self.globalFromSlider.blockSignals(False)
        self.updateGlobalSteps(val, toStep)

    def globalToSpinBoxChanged(self, val):
        fromStep = self.globalFromSpinBox.value()

        self.globalToSlider.blockSignals(True)
        self.globalToSlider.setValue(val)
        self.globalToSlider.blockSignals(False)
        self.updateGlobalSteps(fromStep, val)

    def updateGlobalSteps(self, fromStep, toStep):
        for row in self.rows:
            row.setSteps(fromStep, toStep)

    def updateDataset(self, dataset):
        if dataset is None:
            return

        for row in self.rows:
            row.updateDataset(dataset)

        self.dataset = dataset

    def addRow(self):
        self.rows.append(NeuroVarAvgRowWidget(self.dataset))
        self.mainSnapLayout.addWidget(self.rows[-1])
        self.rows[-1].setMinimumHeight(350)

    def removeRow(self):
        if self.selectedRow is None:
            return

        wdg = self.rows.pop(self.selectedRow)
        self.mainSnapLayout.removeWidget(wdg)
        wdg.setParent(None)
        self.selectedRow = None
        for row in self.rows:
            row.setSelected(False)


class NeuroAnalysis(Ui_NNAnalysis, QtWidgets.QWidget):

    def __init__(self, parent=None):
        super(NeuroAnalysis, self).__init__(parent=parent)
        self.dataset = None

        self.setupUi(self)
        self.setWindowTitle("NeuroPack Analysis Tool")

        self.openDatasetButton.clicked.connect(self._openDataset)

        self.mainStackedWidget.addWidget(NeuroVarSnapWidget(self.dataset))
        self.mainStackedWidget.addWidget(NeuroVarDiffWidget(self.dataset))
        self.mainStackedWidget.addWidget(NeuroVarAvgWidget(self.dataset))
        self.mainStackedWidget.setCurrentIndex(0)

        self.toolsListWidget.currentRowChanged.connect(self.mainStackedWidget.setCurrentIndex)

        self.show()

    def _openDataset(self):
        path = QtWidgets.QFileDialog().getOpenFileName(self, \
                'Open dataset', filter="*.npz")[0]

        if path is None or len(path) == 0:
            return

        try:
            self._updateFromDataset(path)
        except (Exception, ValueError) as exc:
            msgbox = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Critical, \
                    "Error loading file", str(exc), parent=self)
            msgbox.exec_()

    def _updateFromDataset(self, path):
        self.dataset = np.load(path)

        meta = self.dataset['meta']

        trials = pg.siFormat(meta['trials'][0])
        LTP_V = pg.siFormat(meta['LTP_V'][0], suffix='V')
        LTP_pw = pg.siFormat(meta['LTP_pw'][0], suffix='s')
        LTD_V = pg.siFormat(meta['LTD_V'][0], suffix='V')
        LTD_pw = pg.siFormat(meta['LTD_pw'][0], suffix='s')

        self.datasetEdit.setText(os.path.basename(path))
        self.trialsEdit.setText(trials)
        self.ltpVEdit.setText(LTP_V)
        self.ltpPWEdit.setText(LTP_pw)
        self.ltdVEdit.setText(LTD_V)
        self.ltdPWEdit.setText(LTD_pw)

        for i in range(self.mainStackedWidget.count()):
            wdg = self.mainStackedWidget.widget(i)
            wdg.updateDataset(self.dataset)


tags = { 'top': modutils.ModTag("NN", "NeuroPack", None) }
