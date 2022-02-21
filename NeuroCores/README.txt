NeuroPack core library
======================

Place your NeuroPack cores in this folder. All python modules that have their
filename starting with "core_" will be regarded as a distinct core from ArC
ONE. You can also have additional modules if you so require (for example common
code shared by different cores) and as long as they are not prefixed with
"core_" they will be ignored by ArC ONE.

A core file consists of the three functions

* The `init` function is initial setup of the network before any operation is
  done if required. For example you may want to introduce new fields on the
  network object or its state for subsequent use.
* The `neurons` function implements the "evolution" of the network over each
  time step. No training is done in this step.
* The `plast` (plasticity) function implements the "learning" capacity of the
  network for each time step.

If the core produces additional data that need to be saved at the end of the
execution an additional function must be implemented `additional_data` which
returns a dictionary with the parameters that need to be saved in the output
file.

The network
-----------

Common argument for all functions is the network itself (`net`). The network
should have the following fields defined.

* `epochs`: The total number of timesteps for training
* `epochsForTesting`: The total number of timesteps for testing
* `NETSIZE`: The total number of neurons
* `DEPTH`: Depth of the network
* `Ap`, `An`, `a0p`, `a1p`, `a0n`, `a1n`, `tp`, `tn`: memristor parameters
* `rawin`: The raw state of all neurons
* `stimin`: The stimulus input (see NeuroData/motif_stim.txt for an example)
* `ConnMat`: The connectivity matrix (see NeuroData/motif_connmat.txt for an
  example)
* `params`: A dict containing any user defined parameters defined in the base
  configuration excluding `NETSIZE`, `DEPTH`, `Pattern ` that must
  be *always* defined.  By using an alternate base configuration file (see
  NeuroData/Neurobase.json for the base configuration) additional parameters can
  be introduced and will be available under the `params` dict.
* `state`: The internal state of the network (see below). Usually the state of
  the network is what is altered during the `neurons` and `plast` steps.

Network state
-------------
The network object has a `state` field defined. This variable described the
current status of the network and has the following fields defined.

* `weights`: The weights of the neurons for all epochs. This should be altered
  during the plasticity step as it is inherent to training the network.
* `NeurAccum`: Membrane capacitance of the network. Should be updated during
  the `neurons` step.
* `fireCells`: The neuros that should fire during the plasticity step. This is
  introduced from the stimulus file. `fireCells` should be updated during the
  `neurons` step.
* `fireHist`: History of firing neurons. It should be updated during the
  `neurons` step.


