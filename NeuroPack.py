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
from arc1pyqt.VirtualArC import VirtualArC
from arc1pyqt.VirtualArC import pulse as VirtualArCPulse
from arc1pyqt.VirtualArC import read as VirtualArCRead
from arc1pyqt.VirtualArC.parametric_device import ParametricDevice as memristor
from arc1pyqt.Globals import functions
from arc1pyqt import state
HW = state.hardware
APP = state.app
CB = state.crossbar
from arc1pyqt import modutils
from arc1pyqt.modutils import BaseThreadWrapper, BaseProgPanel, \
        makeDeviceList, ModTag

THIS_DIR = os.path.dirname(__file__)
for ui in ['nnanalysis', 'nnvarsnaprow']:

    modutils.compile_ui(os.path.join(THIS_DIR, 'uis', '%s.ui' % ui),
        os.path.join(THIS_DIR, '%s.py' % ui))

from .nnanalysis import Ui_NNAnalysis
from .nnvarsnaprow import Ui_NNVarSnapRow

from . import NeuroCores


def _log(*args, **kwargs):
    if bool(os.environ.get('NNDBG', False)):
        print(*args, file=sys.stderr, **kwargs)

class NetworkState(object):
    """
    NetworkState stores all the information for the state of the
    training process. All history is available.
    """

    def __init__(self, NETSIZE, DEPTH, inputNum, outputNum, epochs, epochsForTesting, temporalCoding_enable, spikeTrain, labelCounter=1):
        super(NetworkState, self).__init__()
        self.weight_addresses = []
        #self.weights = np.array(NETSIZE*[NETSIZE*[epochs*labelCounter*[0.0]]])
        self.weights = np.array((NETSIZE - outputNum) * [(NETSIZE - inputNum) * [epochs * labelCounter * [0.0]]])
        #self.R = np.array((NETSIZE-outputNum) * [(NETSIZE - inputNum) * [4 * epochs * labelCounter * [0.0]]])
        #self.weightsExpected =np.array(NETSIZE*[NETSIZE*[epochs*labelCounter*[0.0]]])
        self.weightsExpected =np.array((NETSIZE-outputNum) * [outputNum * [epochs * labelCounter * [0.0]]])
        #self.weightsError = np.array(NETSIZE*[NETSIZE*[epochs*labelCounter*[0.0]]])
        self.weightsError = np.array((NETSIZE-outputNum) * [outputNum * [epochs * labelCounter * [0.0]]])
        self.NeurAccum = np.zeros(shape=(epochs, (NETSIZE - inputNum)))
        self.NeurAccumForTest = np.zeros(shape=(epochsForTesting, (NETSIZE - inputNum))) #New for test
        self.fireCells = np.array(epochs * labelCounter * [NETSIZE * [0.0]])
        self.fireCellsForTest = np.array(epochsForTesting*labelCounter*[NETSIZE*[0.0]]) #New for test
        self.fireHist = np.array((DEPTH+1) * [NETSIZE * [0.0]])
        self.fireHistForTest = np.array((DEPTH+1)*[NETSIZE*[0.0]]) #New for test
        self.firingCells = np.array(NETSIZE  * [0.0])
        self.firingCellsPseudo = np.array(NETSIZE*[0.0])
        self.outputFlag = 0
        self.neuronFixed = 0
        self.fixedNeuronID = -1
        self.voltMax = np.array((NETSIZE - inputNum)*[0.0])
        self.voltMaxForTest = np.array(NETSIZE*[0.0])
        self.tMax = 0
        self.NeurRecov = np.zeros(shape=(epochs, NETSIZE))
        self.NeurRecovForTest = np.zeros(shape=(epochsForTesting, NETSIZE))
        self.fixedRandomWeights = np.random.rand(NETSIZE, NETSIZE)
        self.spikeTrain_cnt = 0
        self.errorSteps_cnt = 0
        self.errorStepsForTest_cnt = 0
        self.lastSpikeTrain = 0

        self.temporalCoding_enable = temporalCoding_enable
        self.spikeTrain = spikeTrain

        if self.temporalCoding_enable == 1:
            self.errorSteps = epochs // self.spikeTrain
            self.errorStepsForTest = epochsForTesting // self.spikeTrain
        else:
            self.errorSteps = epochs
            self.errorStepsForTest = epochsForTesting
        self.errorList = np.zeros(shape=(self.errorSteps, NETSIZE))
        self.errorListForTest = np.zeros(shape=(self.errorStepsForTest, NETSIZE))


class Network(BaseThreadWrapper):

    """
    This is the abstract represantation of the network. It includes all
    information about the training process, such as potentiation and
    depression potential, number of epochs, batch size, the connection matrix (mapping of
    neurons to devices), the size of the network (`NETSIZE`), number of neurons in each layer,
    and training mode, etc.

    Network parameters are typically loaded from the network configuration file
    which is a JSON file including all the necessary arguments to initialise the
    network. The following arguments are always necessary, regardless whether they
    are used by the core or not

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
    * `Network.stimin`: Forced training stimulus as loaded from the stimulus file
    * `Network.stiminForTesting`: Forced test stimulus as loaded from the stimulus file
    * `Network.testEnable`: The signal indicating whether test is enabled
    * `Network.onlineEnable`: The signal indicating whether online training is enabled
    * `Network.epochs`: Total number of training iterations
    * `Network.epochsForTesting`: Total number of test iterations
    * `Network.NETSIZE`: Size of the network
    * `Network.DEPTH`: Depth of the network
    * `Network.layers`: List of numbers of neurons in each layer
    * `Network.params`: This is a dict containing all JSON parameters as
      picked up from the configuration file.
    * `Network.state`: Complete history of weight and accumulators as well
      as calculated neuron firings. History state population is responsibility
      of the core.
    * `Network.RTolerance` : tolerance ratio in weight updating
    * `Network.pulse(w, b, A, pw)`: Function used to pulse device `(w,b)` with
      a voltage pulse of amplitude `A` and pulse width `pw` (volts and seconds).
      This will immediately return the new resistance of the device.
    * `Network.read(w, b)`: Read-out device `(w,b)` and get its resistance.
    """

    def __init__(self, conn_mat, stimin, stiminForTesting, test_enable, data, params, tsteps, testSteps, core, labelCounter=1):
        super(Network, self).__init__()

        self.ConnMat = conn_mat
        self.stimin = stimin
        self.stiminForTesting = stiminForTesting
        self.testEnable = test_enable

        self.Ap = data["Ap"]
        self.An = data["An"]
        self.a0p = data["a0p"]
        self.a1p = data["a1p"]
        self.a0n = data["a0n"]
        self.a1n = data["a1n"]
        self.tp = data["tp"]
        self.tn = data["tn"]
        self.epochs = data["epochs"]
        self.epochsForTesting = data["epochsForTesting"]
        self.filename = data["fname"]

        print('Ap:%f, An:%f, a0p:%f, a0n:%f, a1p:%f, a1n:%f, tp:%f, tn:%f'%(self.Ap, self.An, self.a0p, self.a0n, self.a1p, self.a1n, self.tp, self.tn))

        # pop the core parameters into distinct fields
        self.NETSIZE = params.pop("NETSIZE")
        self.DEPTH = params.pop("DEPTH")
        self.Pattern_epoch = params.pop("PATTERN_EPOCH")
        self.neuronLock_enable = params.pop('NEURONLOCK_ENABLE')
        self.temporalCoding_enable = params.pop('TEMPORALCODING_ENABLE')
        self.spikeTrain = params.pop('SPIKETRAIN', 1)
        self.layers = params.pop('LAYER')
        self.inputNum = self.layers[0]
        self.outputNum = self.layers[-1]

        self.prefixSum_layers = []
        self.prefixSum_layers.append(self.layers[0])
        for i in range(1, len(self.layers)):
            self.prefixSum_layers.append(self.prefixSum_layers[i - 1] + self.layers[i])

        self.dt = params.pop("dt")
        self.initR = params.pop("INITRES")
        self.variation = params.pop("DEVICEINITVARIATION")
        self.pos_voltOfPulseList = params.pop("POSVOLTOFPULSELIST")
        self.pos_pulsewidthOfPulseList = params.pop("POSPULSEWIDTHOFPULSELIST")
        self.pos_pulseList = list(zip(self.pos_voltOfPulseList, self.pos_pulsewidthOfPulseList))
        self.neg_voltOfPulseList = params.pop("NEGVOLTOFPULSELIST")
        self.neg_pulsewidthOfPulseList = params.pop("NEGPULSEWIDTHOFPULSELIST")
        self.neg_pulseList = list(zip(self.neg_voltOfPulseList, self.neg_pulsewidthOfPulseList))
        # and bundle the rest under self.params
        self.RTolerance = params.pop('RTOLERANCE')
        self.maxSteps = params.pop('MAXUPDATESTEPS')

        self.params = params

        self.tsteps = tsteps
        self.testSteps = testSteps
        self.rawin = np.array(self.NETSIZE*[0])
        self.rawin_pseudo = np.array(self.NETSIZE*[0])
        self.neuronLocked = 0
        self.lockedNeuronID = -1
        self.Vread = HW.conf.Vread

        self.state = NetworkState(self.NETSIZE, self.DEPTH, self.inputNum, self.outputNum, self.epochs, self.epochsForTesting, self.temporalCoding_enable, self.spikeTrain, labelCounter)
        self.plot_counter_trigger = 100
        self.plot_counter = 0
        self.spikeTrainStep = 0

        self.core = self.load_core(core)

    def log(self, *args, **kwargs):
        """ Write to stderr if CTSDBG is set"""
        _log(*args, **kwargs)

    def load_core(self, corename):
        from pkgutil import iter_modules
        import importlib
        basecoremod = 'arc1pyqt.ExtPanels.NeuroPack.NeuroCores'

        for (finder, name, ispkg) in iter_modules(NeuroCores.__path__):
            loader = finder.find_module(name)
            if name == corename:
                mod = importlib.import_module('%s.%s' % (basecoremod, name))
                return mod

    def custom_init(self):
        if not isinstance(HW.ArC, VirtualArC):
            return
        #HW.ArC.crossbar = [[] for x in range(100+1)]
        HW.ArC.crossbar = [[] for x in range(500+1)] # to test the hidden layer
        #for w in range(100+1):
        for w in range(500+1):
            #HW.ArC.crossbar[w].append(0)
            #for b in range(100+1):
            for b in range(500+1):
                mx=memristor(Ap=self.Ap, An=self.An, tp=self.tp, tn=self.tn, a0p=self.a0p, a0n=self.a0n, a1p=self.a1p, a1n=self.a1n)
                mx.initialise(self.initR + (np.random.rand()-0.5)*self.variation)
                HW.ArC.crossbar[w].append(mx)
                #functions.updateHistory(w, b, mx.Rmem, self.Vread, 0.0, 'S R')
                #functions.displayUpdate.cast()


    @BaseThreadWrapper.runner
    def run(self):
        self.disableInterface.emit(True)

        self.log("Reading all devices and initialising weights")

        self.custom_init()
#        f = open("C:/Users/jh1d18/debug_log.txt", "a")
        # For every neuron in the system.
#        for postidx in range(len(self.ConnMat)):
            # For every presynaptic input the neuron receives.
#            for preidx in np.where(self.ConnMat[:, postidx, 0] != 0)[0]:
#                w, b=self.ConnMat[preidx, postidx, 0:2]
#                r = self.read(w,b)
#                f.write('device RS: %f, w: %d, b: %d\n' % (r, w, b))
#        f.close()
#                self.state.weights[preidx, postidx - self.inputNum, 0] = 1.0/self.read(w, b)
                # store device address and neuron ids for easy access in
                # history_panel
#                self.state.weight_addresses.append([[w,b],[preidx,postidx]])
#        self.log("Done.")

        print("Starting Neural Net simulator")

        # Start Neural Net training

        self.core.init(self)

        pattern_epoch_cnt = 0
        print('start training!')
        for t in range(self.tsteps):
            self.rawin = self.state.firingCells
            self.rawinPseudo = self.state.firingCellsPseudo
            if pattern_epoch_cnt == self.Pattern_epoch and self.neuronLock_enable == 1:
                self.neuronLocked = 0
                self.lockedNeuronID = -1
                pattern_epoch_cnt = 0
            else:
                self.neuronLocked = self.state.neuronFixed
                self.lockedNeuronID = self.state.fixedNeuronID
            self.log("---> Time step neuron update in trianing: %d RAWIN: %s STIMIN: %s RAWINPSEUDO: %s" % (t, self.rawin, self.stimin[:, t], self.rawinPseudo))
            self.core.neurons(self, t,  phase = 'training')
            if self.neuronLock_enable:
                pattern_epoch_cnt += 1
            self.rawin = self.state.firingCells
            self.rawinPseudo = self.state.firingCellsPseudo
            self.log("---> Time step synapses update in trianing: %d RAWIN: %s STIMIN: %s RAWINPSEUDO: %s" % (t, self.rawin, self.stimin[:, t], self.rawinPseudo))
            self.core.plast(self, t)
            self.displayData.emit()

        print('testenable in Network: ', self.testEnable)
        if self.testEnable == 1:
            print('start testing!')
            self.weightsForTest = self.state.weights[:, :, self.tsteps - 1]
            for t in range(self.testSteps):
                self.rawin = self.state.firingCells
                self.rawinPseudo = self.state.firingCellsPseudo
                self.log("---> Time step synapses update in trianing: %d RAWIN: %s STIMIN: %s RAWINPSEUDO: %s" % (t, self.rawin, self.stimin[:, t], self.rawinPseudo))
                self.core.neurons(self, t, phase = 'test')
                self.displayData.emit()

        self.log("Final reading of all devices")
        # For every neuron in the system.
        for postidx in range(len(self.ConnMat)):
            # For every presynaptic input the neuron receives.
            for preidx in np.where(self.ConnMat[:, postidx, 0] != 0)[0]:
                w,b=self.ConnMat[preidx, postidx, 0:2]
                self.read(w, b)

        print('fireHistForTest: ', self.state.fireCellsForTest)
        # Save data if so requested
        if self.filename is not None:
            data = {}

            # metadata; this is a numpy structured array
            meta = np.array([(self.epochs, self.epochsForTesting, self.NETSIZE, self.DEPTH, self.inputNum, self.layers, self.Ap, self.An, self.tp, self.tn, self.a0p, self.a0n, self.a1p, self.a1n)],
                    dtype=[('trials', 'u8'), ('trialsForTesting', 'u8'), ('netsize', 'u8'), ('depth', 'u8'), ('inputNum', 'u8'), ('layers', 'O'), ('Ap', 'f4'), ('An', 'f4'),
                        ('tp', 'f4'), ('tn', 'f4'), ('a0p', 'f4'), ('a0n', 'f4'), ('a1p', 'f4'), ('a1n', 'f4')])
            data['meta'] = meta

            # standard data first

            # all weights
            data['weights'] = self.state.weights
            data['weightsExpected'] = self.state.weightsExpected
            data['weightsError'] = self.state.weightsError
            # calculated stimuli for each step
            data['stimulus'] = self.stimin
            data['stimulusForTest'] = self.stiminForTesting
            # history of cells that have fired
            data['fires'] = self.state.fireCells.T
            data['firesForTest'] = self.state.fireCellsForTest.T
            # accumulator snapshots
            data['accumulator'] = self.state.NeurAccum.T
            data['accumulatorForTest'] = self.state.NeurAccumForTest.T
            data['membraneRecoveryVariable'] = self.state.NeurRecov.T
            data['membraneRecoveryVariableForTest'] = self.state.NeurRecovForTest.T
            # error between outputs and labels. No error for unsupervised learning
            data['error'] = self.state.errorList
            data['errorForTest'] = self.state.errorListForTest
            #data['R'] = self.state.R

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

        #Mnow = HW.ArC.read_one(w, b)
        Mnow = VirtualArCRead(HW.ArC.crossbar, w, b)
        self.sendData.emit(w, b, Mnow, self.Vread, 0, \
                'S R%d V=%.1f' % (HW.conf.readmode, HW.conf.Vread))
        self.updateTree.emit(w, b)

        return Mnow

    def pulse(self, w, b, A, pw):
        # apply a pulse and return
        # can instead apply any voltage series
        self.highlight.emit(w,b)

        #Mnow = HW.ArC.pulseread_one(w, b, A, pw)
        VirtualArCPulse(HW.ArC.crossbar, w, b, A, pw, self.dt)
        Mnow = VirtualArCRead(HW.ArC.crossbar, w, b)

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
        self.test_file_fname = None
        self.output_file_fname = None
        self.test_enable = 0
        self.initUI()

        fname = os.path.join(THIS_DIR, "NeuroData", "NeuroBase.json")
        params = self.load_base_conf(os.path.join(THIS_DIR, "NeuroData",\
            "NeuroBase.json"))
        self.apply_base_conf(params, os.path.basename(fname), fname)

    # def execute(self, wrapper, entrypoint=None, deferredUpdate=False, signals=True):
        # """
        # This function schedules a wrapper for execution taking care of the
        # standard signals. The wrapped action (`wrapper`) will be passed
        # along a thread which will call the `entrypoint` function of
        # `wrapper`. If `entrypoint` is None the default `wrapper.run`
        # entrypoint will be used. Argument `deferredUpdate` prevents the history
        # tree from updating until the thread operation has finished. This can
        # be useful in situations where multiple different devices are used or
        # when a module uses many individual operations that would otherwise
        # trigger a tree update (for instance hundreds of reads/pulses over
        # ten different devices).
        # """
        # if (HW.ArC is None) or (self.thread is not None):
            # return

        # if entrypoint is None:
            # entrypoint = wrapper.run

        # self.threadWrapper = wrapper
        # self.thread = QtCore.QThread()

        # # When deferring tree updates store current point in history for the
        # # whole crossbar. Once the operation is finished the history tree will
        # # then be populated starting from this point in history
        # if deferredUpdate:
            # for (r, row) in enumerate(CB.history):
                # for (c, col) in enumerate(row):
                    # self._deferredUpdates['%d%d' % (r, c)] = (r, c, len(col))

        # self.threadWrapper.moveToThread(self.thread)
        # self.thread.started.connect(entrypoint)
        # self.threadWrapper.finished.connect(self.thread.quit)
        # if signals:
            # #self.threadWrapper.sendData.connect(functions.updateHistory)
            # #self.threadWrapper.highlight.connect(functions.cbAntenna.cast)
            # #self.threadWrapper.displayData.connect(functions.displayUpdate.cast)
            # if not deferredUpdate:
                # self.threadWrapper.updateTree.connect(\
                    # functions.historyTreeAntenna.updateTree.emit)
        # self.threadWrapper.disableInterface.connect(functions.interfaceAntenna.cast)
        # self.thread.finished.connect(partial(self._onThreadFinished, deferredUpdate, signals))
        # self.thread.start()

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
        self.base_conf_filename = QtWidgets.QLabel("Load Base conf.")
        gridLayout.addWidget(self.push_load_base_conf, 0, 1)
        gridLayout.addWidget(self.base_conf_filename, 0, 0)

        self.push_load_conn_matrix = QtWidgets.QPushButton("Load Conn. Matrix")
        self.push_load_conn_matrix.clicked.connect(self.open_conn_matrix)
        self.matrix_filename = QtWidgets.QLabel("Load Conn. Matrix")
        gridLayout.addWidget(self.push_load_conn_matrix, 1, 1)
        gridLayout.addWidget(self.matrix_filename, 1, 0)

        self.push_load_stim_file = QtWidgets.QPushButton("Load Stim. File")
        self.push_load_stim_file.clicked.connect(self.open_stim_file)
        self.stim_filename = QtWidgets.QLabel("Load Stim. File")
        gridLayout.addWidget(self.push_load_stim_file, 2 ,1)
        gridLayout.addWidget(self.stim_filename, 2, 0)

        self.check_test_file = QtWidgets.QCheckBox("Load Test File")
        self.check_test_file.clicked.connect(self.check_test_file_clicked)
        self.test_filename = QtWidgets.QPushButton("Load Test File")
        self.test_filename.clicked.connect(self.open_test_file)
        self.test_filename.setEnabled(False)
        gridLayout.addWidget(self.check_test_file, 3, 0)
        gridLayout.addWidget(self.test_filename, 3, 1)

        self.check_save_data = QtWidgets.QCheckBox("Save to:")
        self.check_save_data.clicked.connect(self.check_save_data_clicked)
        self.push_save_filename = QtWidgets.QPushButton("No file selected")
        self.push_save_filename.clicked.connect(self.load_output_file)
        self.push_save_filename.setEnabled(False)
        gridLayout.addWidget(self.check_save_data, 5, 0)
        gridLayout.addWidget(self.push_save_filename, 5, 1)

        self.push_show_analysis_tool = QtWidgets.QPushButton("Start analysis tool")
        self.push_show_analysis_tool.clicked.connect(self.startAnalysisTool)
        gridLayout.addWidget(self.push_show_analysis_tool, 9, 0, 1, 2)

        ####################################################################

        ################################################## CORES ###########

        self.rulesCombo = QtWidgets.QComboBox()
        for _, name, is_pkg in pkgutil.iter_modules(NeuroCores.__path__):
            if not is_pkg and name.startswith("core_"):
                self.rulesCombo.addItem(name.replace("core_", ""), name)

        gridLayout.addWidget(QtWidgets.QLabel("Network core:"), 4, 0)
        gridLayout.addWidget(self.rulesCombo, 4, 1)

        ####################################################################

        leftLabels=['Trials for training',\
                    'Trials for testing']
        self.leftEdits=[]

        rightLabels=['Ap',\
                    'An',\
                    'a0p',\
                    'a0n',\
                    'a1p',\
                    'a1n',\
                    'tp',\
                    'tn'
                    ]
        self.rightEdits=[]

        leftInit=  ['10000',\
                    '10000']

        rightInit = ['0.21388644421061628',\
                    '-0.813018367268805',\
                    '37086.67218413958',\
                    '43430.02023698205',\
                    '-20193.23957579438',\
                    '34332.85303661032',\
                    '1.6590989889370842',\
                    '1.5148294827972748'
                    ]

        #setup a line separator
        lineLeft = QtWidgets.QFrame()
        lineLeft.setFrameShape(QtWidgets.QFrame.VLine);
        lineLeft.setFrameShadow(QtWidgets.QFrame.Raised);
        lineLeft.setLineWidth(1)

        gridLayout.addWidget(lineLeft, 0, 2, 11, 1)

        for i in range(len(leftLabels)):
            lineLabel=QtWidgets.QLabel()
            lineLabel.setText(leftLabels[i])
            gridLayout.addWidget(lineLabel, i+6,0)

            lineEdit=QtWidgets.QLineEdit()
            lineEdit.setText(leftInit[i])
            lineEdit.setValidator(isFloat)
            self.leftEdits.append(lineEdit)
            gridLayout.addWidget(lineEdit, i+6,1)

        for i in range(len(rightLabels)):
            lineLabel=QtWidgets.QLabel()
            lineLabel.setText(rightLabels[i])
            gridLayout.addWidget(lineLabel, i,4)

            lineEdit=QtWidgets.QLineEdit()
            lineEdit.setText(rightInit[i])
            lineEdit.setValidator(isFloat)
            self.rightEdits.append(lineEdit)
            gridLayout.addWidget(lineEdit, i,5)

        self.Ap=float(self.rightEdits[0].text())
        self.An=float(self.rightEdits[1].text())
        self.a0p=float(self.rightEdits[2].text())
        self.a0n=float(self.rightEdits[3].text())
        self.a1p=float(self.rightEdits[4].text())
        self.a1n=float(self.rightEdits[5].text())
        self.tp=float(self.rightEdits[6].text())
        self.tn=float(self.rightEdits[7].text())
        self.epochs=int(self.leftEdits[0].text())
        self.epochsForTesting=int(self.leftEdits[1].text())

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
            "Ap": float(self.rightEdits[0].text()), \
            "An": float(self.rightEdits[1].text()), \
            "a0p": float(self.rightEdits[2].text()),\
            "a0n": float(self.rightEdits[3].text()),\
            "a1p": float(self.rightEdits[4].text()),\
            "a1n": float(self.rightEdits[5].text()),\
            "tp": float(self.rightEdits[6].text()),\
            "tn": float(self.rightEdits[7].text()),\
            "epochs": int(self.leftEdits[0].text()),\
            "epochsForTesting": int(self.leftEdits[1].text()),\
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
        testSteps = data["epochsForTesting"]

        # Reload the stimulus file to account for any changes in the epochs
        # Could possibly check if the field is "tainted" before loading to
        # avoid accessing the file again
        self.stimin = self.load_stim_file(self.stim_file_fname, \
            params["NETSIZE"], data["epochs"])
        self.stiminForTesting = self.load_test_file(self.test_file_fname, \
            params["NETSIZE"], data["epochsForTesting"])

        if HW.ArC is not None:
            coreIdx = self.rulesCombo.currentIndex()
            coreName = self.rulesCombo.itemData(coreIdx)
            print('test_enable before calling network:', self.test_enable)
            network = Network(self.ConnMat, self.stimin, self.stiminForTesting, self.test_enable, data, params, \
                tsteps, testSteps, coreName, self.labelCounter)
            self.execute(network, network.run, True)

    def load_base_conf(self, fname):
        return json.load(open(fname))

    def open_base_conf(self):
        path = QtCore.QFileInfo(QtWidgets.QFileDialog().getOpenFileName(self,\
            'Open base configuration', THIS_DIR, filter="*.json")[0])
        name = path.fileName()

        try:
            data = self.load_base_conf(path.filePath())
            for x in ["NETSIZE", "DEPTH"]:#!!!!!!!!!!!!
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

        if self.test_file_fname is not None:
            data = self.gather_data()
            stims = self.load_test_file(self.test_file_fname, params["NETSIZE"],\
                data["epochsForTesting"])
            self.apply_test_file(stims)

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

    def check_test_file_clicked(self, checked):
        self.test_filename.setEnabled(checked)
        if checked == True:
            self.test_enable = 1
        else:
            self.test_enable = 0
        print('test_enable in checked_save_data_clicked:', self.test_enable)

    def load_test_file(self, fname, NETSIZE, epochsForTesting):
        _log("Allocating test stimin")
        _log('epochsForTesting when loading test file:', epochsForTesting)
        stiminForTesting = np.array(NETSIZE*[epochsForTesting*[0]])
        if fname is not None:
            _log(fname)
            with open(fname, 'r') as f:
                _log("File opened")
                for line in f:
                    line = line.strip()
                    if (line[0] != "\n") and (line[0] != "#"):
                        # split into timestamp - list of neurons IDs scheduled to spike
                        timestamp, neuronIDs = re.split("\s*-\s*", line)
                        timestamp = int(timestamp)
                        if timestamp >= epochsForTesting:
                            break
                        _log(timestamp, neuronIDs)
                        # split the string into an int list of neurons
                        spikeNeuronID = [int(x) - 1 for x in re.split("\s*,\s*", neuronIDs.strip())]
                        for i, spiker in enumerate(spikeNeuronID):
                            stiminForTesting[spiker, timestamp] = 1
        return stiminForTesting

    def open_test_file(self):
        _log("Loading test file...")

        params = self.gather_params()
        data = self.gather_data()

        path = QtCore.QFileInfo(QtWidgets.QFileDialog().getOpenFileName(self,\
            'Open test file', THIS_DIR, filter="*.txt")[0])
        name = path.fileName()

        error = False
        try:
            res = self.load_test_file(path.filePath(), params["NETSIZE"], \
                data["epochsForTesting"])
        except Exception as exc:
            _log(exc)
            error = True

        #print(self.stimin)

        if error:
            self.stiminForTesting = np.array(params["NETSIZE"]*[data["epochsForTesting"]*[0]])
            errMessage = QtWidgets.QMessageBox()
            errMessage.setText("Invalid network test file! Possible problem with syntax.")
            errMessage.setIcon(QtWidgets.QMessageBox.Critical)
            errMessage.setWindowTitle("Error")
            errMessage.exec_()
        else:
            self.apply_test_file(res, name, path.filePath())

        _log("done")

    def apply_test_file(self, stim, name=None, path=None):
        self.stiminForTesting = stim
        if name is not None:
            self.test_filename.setText(name)
        if path is not None:
            self.test_file_fname = str(path)

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
        self.currentIdx = None

        self.updateDataset(dataset)
        self.setSelected(self.selected)
        self.stepSlider.valueChanged.connect(self.sliderChanged)
        self.stepSpinBox.valueChanged.connect(self.stepSpinBoxChanged)
        self.NeuronSpinBox.valueChanged.connect(self.NeuronSpinBoxChanged)
        self.LayerSpinBox.valueChanged.connect(self.LayerSpinBoxChanged)
        self.RowNumSpinBox.valueChanged.connect(self.RowNumSpinBoxChanged)
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

    def NeuronSpinBoxChanged(self, val):
        self.NeuronSpinBox.blockSignals(True)
        self.NeuronSpinBox.setValue(val)
        self.NeuronSpinBox.blockSignals(False)
        currentStep = self.stepSpinBox.value()
        self.updatePlotToStep(currentStep)

    def LayerSpinBoxChanged(self, val):
        self.LayerSpinBox.blockSignals(True)
        self.LayerSpinBox.setValue(val)
        self.LayerSpinBox.blockSignals(False)
        currentStep = self.stepSpinBox.value()
        self.updatePlotToStep(currentStep)

    def RowNumSpinBoxChanged(self, val):
        self.RowNumSpinBox.blockSignals(True)
        self.RowNumSpinBox.setValue(val)
        self.RowNumSpinBox.blockSignals(False)
        currentStep = self.stepSpinBox.value()
        self.updatePlotToStep(currentStep)

    def _updateGraph(self, data):

        inputNeuronNum = int(self.dataset['meta']['inputNum'][0])

        plotArgs = {'pen': pg.mkPen(color=(80, 129, 204), width=1, style=QtCore.Qt.SolidLine), 'symbolPen': None, 'symbolBrush': (80, 129, 204), \
                'symbol':'+'}
        colors = [(185,119,165), (229,129,158), (16,749,453), (255,147,141), (255,175,120), (249,248,113)]

        if self.idx != self.currentIdx:
            self.currentIdx = self.idx
            self._clearGraphs()
            if self.idx == 0:    # weight mapping
                wdg = pg.ImageView()
                wdg.ui.menuBtn.hide()
                wdg.ui.roiBtn.hide()
                wdg.setImage(data.T[self.step])
                cmap = pg.ColorMap(pos=np.linspace(0.0, 1.0, 6), color = colors)
                wdg.setColorMap(cmap)
            elif self.idx == 1: # weight for each layer
                wdg = pg.ImageView()
                wdg.ui.menuBtn.hide()
                wdg.ui.roiBtn.hide()
                wdg.setImage(data[self.prefixSum_layerList[self.layerIdx - 1] : self.prefixSum_layerList[self.layerIdx], (self.neuronIdx - inputNeuronNum), self.step].reshape((self.RowNumIdx, -1)))
                cmap = pg.ColorMap(pos=np.linspace(0.0, 1.0, 6), color = colors)
                wdg.setColorMap(cmap)
            elif self.idx == 2:    # accumulator for training
                wdg = pg.PlotWidget()
                wdg.plot(np.arange(len(data[0])), data[self.neuronIdx], **plotArgs)
            elif self.idx == 3:    # accumulator for test
                wdg = pg.PlotWidget()
                wdg.plot(np.arange(len(data[0])), data[self.neuronIdx], **plotArgs)
            elif self.idx == 4:    # neuron recovery variable for training, only for Izhikevich neuron model
                wdg = pg.PlotWidget()
                wdg.plot(np.arange(len(data[0])), data[self.neuronIdx], **plotArgs)
            elif self.idx == 5:    # neuron recovery variable for test, only for Izhikevich neuron model
                wdg = pg.PlotWidget()
                wdg.plot(np.arange(len(data[0])), data[self.neuronIdx], **plotArgs)
            elif self.idx == 6:    # fire history for training
                wdg = pg.PlotWidget()
                wdg.plot(np.arange(len(data[0])) + 1, data[self.neuronIdx], **plotArgs)
            elif self.idx == 7:    # fire history for test
                wdg = pg.PlotWidget()
                wdg.plot(np.arange(len(data[0])) + 1, data[self.neuronIdx], **plotArgs)
                print('neuron index: ', self.neuronIdx)
                print('fireHist: ', data[self.neuronIdx])
            elif self.idx == 8:     # input stimulus for training
                wdg = pg.ImageView()
                wdg.ui.menuBtn.hide()
                wdg.ui.roiBtn.hide()
                wdg.setImage(data[:inputNeuronNum, self.step].reshape((self.RowNumIdx, -1)))
            elif self.idx == 9:     # input stimulus for test
                wdg = pg.ImageView()
                wdg.ui.menuBtn.hide()
                wdg.ui.roiBtn.hide()
                wdg.setImage(data[:inputNeuronNum, self.step].reshape((self.RowNumIdx, -1)))
            elif self.idx == 10:     # error for training
                wdg = pg.PlotWidget()
                wdg.plot(np.arange(len(data)), data[:, self.neuronIdx], **plotArgs)
            elif self.idx == 11:    # error for test
                wdg = pg.PlotWidget()
                wdg.plot(np.arange(len(data)), data[:, self.neuronIdx], **plotArgs)
            elif self.idx == 12:    # expected weights for each layer
                wdg = pg.ImageView()
                wdg.ui.menuBtn.hide()
                wdg.ui.roiBtn.hide()
                wdg.setImage(data[self.prefixSum_layerList[self.layerIdx - 1] : self.prefixSum_layerList[self.layerIdx], (self.neuronIdx - inputNeuronNum), self.step].reshape((self.RowNumIdx, -1)))
                cmap = pg.ColorMap(pos=np.linspace(0.0, 1.0, 6), color = colors)
                wdg.setColorMap(cmap)
            elif self.idx == 13:    # weight error for each layer
                wdg = pg.ImageView()
                wdg.ui.menuBtn.hide()
                wdg.ui.roiBtn.hide()
                wdg.setImage(data[self.prefixSum_layerList[self.layerIdx - 1] : self.prefixSum_layerList[self.layerIdx], (self.neuronIdx - inputNeuronNum), self.step].reshape((self.RowNumIdx, -1)))
                cmap = pg.ColorMap(pos=np.linspace(0.0, 1.0, 6), color = colors)
                wdg.setColorMap(cmap)
            self.graphHolderLayout.addWidget(wdg)
            self.plotWidget = wdg
        else:
            if self.idx == 0:    # weight mapping
                self.plotWidget.setImage(data.T[self.step])
            elif self.idx == 1:     # weight for each layer
                self.plotWidget.setImage(data[self.prefixSum_layerList[self.layerIdx - 1] : self.prefixSum_layerList[self.layerIdx], (self.neuronIdx - inputNeuronNum), self.step].reshape((self.RowNumIdx, -1)))
            elif self.idx == 2:     # accumulator for training
                self.plotWidget.plot(np.arange(len(data[0])), data[self.neuronIdx], clear=True, **plotArgs)
            elif self.idx == 3:     # accumulator for test
                self.plotWidget.plot(np.arange(len(data[0])), data[self.neuronIdx], clear=True, **plotArgs)
            elif self.idx == 4:     # neuron recovery variable for training, only for Izhikevich neuron model
                self.plotWidget.plot(np.arange(len(data[0])), data[self.neuronIdx], clear=True, **plotArgs)
            elif self.idx == 5:     # neuron recovery variable for test, only for Izhikevich neuron model
                self.plotWidget.plot(np.arange(len(data[0])), data[self.neuronIdx], clear=True, **plotArgs)
            elif self.idx == 6:     # fire history for training
                self.plotWidget.plot(np.arange(len(data[0])) + 1, data[self.neuronIdx], clear=True, **plotArgs)
            elif self.idx == 7:     # fire history for test
                self.plotWidget.plot(np.arange(len(data[0])) + 1, data[self.neuronIdx], clear=True, **plotArgs)
                print('neuron index: ', self.neuronIdx)
                print('fireHist: ', data[self.neuronIdx])
            elif self.idx == 8:     # input stimulus for training
                self.plotWidget.setImage(data[:inputNeuronNum, self.step].reshape((self.RowNumIdx, -1)))
            elif self.idx == 9:     # input stimulus for test
                self.plotWidget.setImage(data[:inputNeuronNum, self.step].reshape((self.RowNumIdx, -1)))
            elif self.idx == 10:     # error for training
                self.plotWidget.plot(np.arange(len(data)), data[:, self.neuronIdx], clear=True, **plotArgs)
            elif self.idx == 11:     # error for test
                self.plotWidget.plot(np.arange(len(data)), data[:, self.neuronIdx], clear=True, **plotArgs)
            elif self.idx == 12:    # expected weights for each layer
                self.plotWidget.setImage(data[self.prefixSum_layerList[self.layerIdx - 1] : self.prefixSum_layerList[self.layerIdx], (self.neuronIdx - inputNeuronNum), self.step].reshape((self.RowNumIdx, -1)))
            elif self.idx == 13:    # weight error for each layer
                self.plotWidget.setImage(data[self.prefixSum_layerList[self.layerIdx - 1] : self.prefixSum_layerList[self.layerIdx], (self.neuronIdx - inputNeuronNum), self.step].reshape((self.RowNumIdx, -1)))

    def updatePlotToStep(self, step):
        self.idx = self.variableSelectionCombo.currentIndex()
        self.step = step
        self.neuronIdx = self.NeuronSpinBox.value()
        self.layerIdx = self.LayerSpinBox.value()
        self.RowNumIdx = self.RowNumSpinBox.value()
        self.layerList = self.dataset['meta']['layers'][0]
        self.prefixSum_layerList = []
        self.prefixSum_layerList.append(0)
        for i in range(len(self.layerList)):
            self.prefixSum_layerList.append(self.prefixSum_layerList[i - 1] + self.layerList[i])

        if self.idx == 0 or self.idx == 1:    # weight mapping
            data = self.dataset['weights']
        elif self.idx == 2:  # membrane volt for training
            data = self.dataset['accumulator']
        elif self.idx == 3:  # membrane volt for test
            data = self.dataset['accumulatorForTest']
        elif self.idx == 4:  # membrane Recovery Variable for training
            data = self.dataset['membraneRecoveryVariable']
        elif self.idx == 5:  # membrane Recovery Variable for test
            data = self.dataset['membraneRecoveryVariableForTest']
        elif self.idx == 6: # fire history for training
            data = self.dataset['fires']
        elif self.idx == 7: # fire history for test
            data = self.dataset['firesForTest']
            print('fireCellsForTest in GUI:', self.dataset['firesForTest'])
        elif self.idx == 8:  # stimulus for training
            data = self.dataset['stimulus']
        elif self.idx == 9:  # stimulus for test
            data = self.dataset['stimulusForTest']
        elif self.idx == 10: # error for training
            data = self.dataset['error']
        elif self.idx == 11: # error for test
            data = self.dataset['errorForTest']
        elif self.idx == 12: # expected weight for each layer
            data = self.dataset['weightsExpected']
        elif self.idx == 13: # weight error for each layer
            data = self.dataset['weightsError']
        self._updateGraph(data)

    def updateDataset(self, dataset):

        if dataset is None:
            return

        self.stepSlider.setMinimum(0)
        self.stepSlider.setMaximum(int(dataset['meta']['trials'][0])-1)

        self.stepSpinBox.setMinimum(0)
        self.stepSpinBox.setMaximum(int(dataset['meta']['trials'][0])-1)

        self.NeuronSpinBox.setMinimum(int(self.dataset['meta']['inputNum'][0]))
        self.NeuronSpinBox.setMaximum(int(dataset['meta']['netsize'][0])-1)

        self.LayerSpinBox.setMinimum(1)
        self.LayerSpinBox.setMaximum(len(dataset['meta']['layers'][0])-1)

        self.RowNumSpinBox.setMinimum(1)

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


class NeuroAnalysis(Ui_NNAnalysis, QtWidgets.QWidget):

    def __init__(self, parent=None):
        super(NeuroAnalysis, self).__init__(parent=parent)
        self.dataset = None

        self.setupUi(self)
        self.setWindowTitle("NeuroPack Analysis Tool")

        self.openDatasetButton.clicked.connect(self._openDataset)

        self.rows =  []
        self.selectedRow = None
        self.AddRowBotton.clicked.connect(self.addRow)
        self.RemoveRowBotton.clicked.connect(self.removeRow)
        self.checkBox.stateChanged.connect(self.lockStepsChecked)
        self.globalStepSlider.valueChanged.connect(self.stepSliderChanged)
        self.globalStepSpinBox.valueChanged.connect(self.stepSpinBoxChanged)

        self.show()

    def mousePressEvent(self, evt):
        for (idx, row) in enumerate(self.rows):
            if row.underMouse():
                self.selectedRow = idx
            row.setSelected(row.underMouse())

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
        self.dataset = np.load(path, allow_pickle=True)

        meta = self.dataset['meta']

        Training = pg.siFormat(meta['trials'][0])
        Test = pg.siFormat(meta['trialsForTesting'][0])
        Ap = pg.siFormat(meta['Ap'][0])
        An = pg.siFormat(meta['An'][0])
        tp = pg.siFormat(meta['tp'][0])
        tn = pg.siFormat(meta['tn'][0])
        a0p = pg.siFormat(meta['a0p'][0])
        a0n = pg.siFormat(meta['a0n'][0])
        a1p = pg.siFormat(meta['a1p'][0])
        a1n = pg.siFormat(meta['a1n'][0])

        self.datasetEdit.setText(os.path.basename(path))
        self.TrainingEdit.setText(Training)
        self.testEdit.setText(Test)
        self.ApEdit.setText(Ap)
        self.AnEdit.setText(An)
        self.tpEdit.setText(tp)
        self.tnEdit.setText(tn)
        self.a0pEdit.setText(a0p)
        self.a0nEdit.setText(a0n)
        self.a1pEdit.setText(a1p)
        self.a1nEdit.setText(a1n)

    def lockStepsChecked(self):
        checked = self.checkBox.isChecked()
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


tags = { 'top': modutils.ModTag("NN", "NeuroPack", None) }
