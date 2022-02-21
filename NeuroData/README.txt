NeuroPack core library

------------------------------------------------------------------------------
All necessary configuration parameters in NeuroBase.json file:

* `NETSIZE`: The total number of neurons
* `DEPTH`: Depth of the network
* `SPIKETRAIN`: if spike train is enabled as inputs. 
* `Pattern_epoch`: the number of spikes regarded as one input if spike train is enabled. Currently used in tempotron core.
* `temporalCoding_enable`: if temporal coding is enabled.
* `LAYER`: the number of layers
* `dt`: time step used in Izhkevich neuron core.
* `INITRES`: memristors initialised resistance states
* `DEVICEINITVAIATION`: memristor initialisation variation
* `POSVOLTOFPULSELIST`: magnitudes of pulse options for positive pulses
* `POSPULSEWIDTHOFPULSELIST`: pulsewidth of pulse options for positive pulses
* `NEGVOLTOFPULSELIST`: magnitudes of pulse options for negative pulses
* `NEGPULSEWIDTHOFPULSELIST`: pulsewidth of pulse options for negative pulses
* `MAXUPDATESTEPS`: maximum update step number for weight updating
* `RTOLERANCE`: R tolerance. Defined as (R_real - R_exp)/R_exp 

------------------------------------------------------------------------------
connectivity matrix format:
# PREID, POSTID, W, B, TYPE
# TYPE can be either +1 (excitatory) or -1 (inhibitory)

------------------------------------------------------------------------------
stimuli file format:
# timestep - neuron ID that fires, neuron ID that fires, neuron ID that fires, ...., neuron ID that fires