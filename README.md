# NeuroPack
Paper available here: [NeuroPack: An Algorithm-level Python-based Simulator for Memristor-empowered Neuro-inspired Computing](https://arxiv.org/abs/2201.03339)

`experimentalResults.npz` gives experimental results shown in the paper.

NeuroPack is an algorithm-level simulator for memrsitor-based neuromorphic designs. It incorporates an empirical memristor model [here](https://arxiv.org/abs/1703.01167) to simulate how memristor states change during an online-training process.
To get memristor fitting parameters, please use [ArC One](https://github.com/arc-instruments/arc1_pyqt) instrument module 'ParameterFit'.

---
#### Components
 - `NeuroPack.py` implements the main panel and the built-in analysis tool for NeuroPack
 - `NeuroCores/core_X.py` implements the neuron model and the learning rule

---
#### Input files
 - Configuration file: `NeuroData/XX.json` is an input file that users need to provide for setting configuration
 - Matrix Connectivity file: `NeuroData/XX.txt` is an input file that defines how neurons are connected and which memristor is mapped to the connectivity.
 - Stimuli file for training: `NeuroData/XX.txt` is an input file that provides input spikes and output labels (for supervised learning only) for training.
 - Stimuli file for testing (optional): `NeuroData/XX.txt` is an input file that provide input spikes and output labels (for supervised learning only) for test.

---
#### Install and run
 - Please install the updated [Arc One](https://github.com/hjq310/arc1_pyqt) interface with all required packages.
 - Download NeuroPack and put it in the following directory:
C:/Users/AppData/Roaming/arc1pyqt\ProgPanels
 - If you use Windows, open command prompt and go to the path where you install arc1_pyqt.
 - Use the following commands to run Arc One interface:

 `SET NNDBG=1` if you want to print out debug information in command prompt.

 `python setup.py build`

 `python run.py`
 - Now ArC One interface pops up. Select 'VirtualArC' from the portal dropdown list, and click 'connect'.
 - Click 'Read All'. Now the state indicator changes to 'Ready'.
 - Select 'NeuroPack' from module dropdown list, and click 'Add'.
 - Now NeuroPack main panel should pop up. Add all necessary input files and click 'Training network' to run.
---
#### Example usage
If you want to reproduce the results showcased in the paper 'NeuroPack: An Algorithm-level Python-based Simulator for Memristor-empowered Neuro-inspired Computing', please do the following:
 - Click 'NeuroBase.json' and select 'MNIST_LIF.json'.
 - Click 'Load Conn. Matrix' and select 'MNIST_LIF_connmat.txt'.
 - Click 'Load Stim. File' and select 'MNIST_LIF_stim.txt'.
 - Tick 'Load test file', click and select 'MNIST_LIF_test_stim.txt'.
 - Tick 'Save to', click 'Load test file', and create a file to store inference results.
 - Select 'LIF_supervisedLearning_wta' for 'Network core'.
 - Press 'Train Network'
---
#### Citation
```
@misc{huang2022neuropack,
      title={NeuroPack: An Algorithm-level Python-based Simulator for Memristor-empowered Neuro-inspired Computing},
      author={Jinqi Huang and Spyros Stathopoulos and Alex Serb and Themis Prodromakis},
      year={2022},
      eprint={2201.03339},
      archivePrefix={arXiv},
      primaryClass={cs.ET}
}
```
