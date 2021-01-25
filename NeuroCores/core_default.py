import numpy as np

# A NeuroPack core implements plasticity events and updates
# Essentialy NeuroPack will call the following functions
#
# init(network)
# for trial in trials:
#   neurons(network, step)
#   plast(network, step)

# This particular core implements a LIF network, the neurons function
# calculates which neurons are set to fire and the plast function
# implements plasticity event propagation.

# neurons and plast both take two arguments. The first one is the
# network itself (see `NeuroPack.Network`) and the second is the
# timestep.

# This core requires `NeuroData/SevenMotif.json`.

def normalise_weight(net, w):
    PCEIL = 1.0/net.params['PFLOOR']
    PFLOOR = 1.0/net.params['PCEIL']

    val = net.params['WEIGHTSCALE']*(float(w) - PFLOOR)/(PCEIL - PFLOOR)

    # Clamp weights in-between 0.0 and 1.0
    if val < 0.0:
        return 0.0
    elif val > 1.0:
        return 1.0
    else:
        return val


def init(net):
    # Renormalise weights if needed
    if not net.params.get('NORMALISE', False):
        return

    for postidx in range(len(net.ConnMat)):
        # For every presynaptic input the neuron receives.
        for preidx in np.where(net.ConnMat[:, postidx, 0] != 0)[0]:
            old_weight = net.state.weights[preidx, postidx, 0]
            new_weight = normalise_weight(net, old_weight)
            net.state.weights[preidx, postidx, 0] = new_weight


def neurons(net, time):

    rawin = net.rawin # Raw input
    stimin = net.stimin[:, time] # Stimulus input for current timestep

    full_stim = np.bitwise_or([int(x) for x in rawin], [int(x) for x in stimin])
    net.log("**** FULL_STIM = ", full_stim)

    if time > 0:
        # if this isn't the first step copy the accumulators
        # from the previous step onto the new one
        net.state.NeurAccum[time] = net.state.NeurAccum[time-1]

    # reset the accumulators of neurons that have already fired
    for (idx, v) in enumerate(full_stim):
        if v != 0:
            net.state.NeurAccum[time][idx] = 0.0

    # For this example we'll make I&F neurons - if changing this file a back-up
    # is strongly recommended before proceeding.

    #  -FIX- implementing 'memory' between calls to this function.
    # NeurAccum = len(net.ConnMat)*[0] #Define neuron accumulators.
    # Neurons that unless otherwise dictated to by net or ext input will
    # fire.
    wantToFire = len(net.ConnMat)*[0]

    # Gather/define other pertinent data to function of neuron.
    leakage = net.params.get('LEAKAGE', 1.0)
    bias = np.array(len(net.ConnMat)*[leakage]) #No active biases.

    # STAGE I: See what neurons do 'freely', i.e. without the constraints of
    # WTA or generally other neurons' activities.
    for postidx in range(len(net.state.NeurAccum[time])):
        # Unconditionally add bias term
        net.state.NeurAccum[time][postidx] += bias[postidx]
        if net.state.NeurAccum[time][postidx] < 0.0:
            net.state.NeurAccum[time][postidx] = 0.0

        #For every presynaptic input the neuron receives.
        for preidx in np.where(net.ConnMat[:, postidx, 0] != 0)[0]:

            # Excitatory case
            if net.ConnMat[preidx, postidx, 2] > 0:
                # net.log("Excitatory at %d %d" % (preidx, postidx))
                # Accumulator increases as per standard formula.
                net.state.NeurAccum[time][postidx] += \
                    full_stim[preidx] * net.state.weights[preidx, postidx, time]

                net.log("POST=%d PRE=%d NeurAccum=%g full_stim=%g weight=%g" % \
                    (postidx, preidx, net.state.NeurAccum[time][postidx], \
                     full_stim[preidx], net.state.weights[preidx, postidx, time]))

            # Inhibitory case
            elif net.ConnMat[preidx, postidx, 2] < 0:
                # Accumulator decreases as per standard formula.
                net.state.NeurAccum[time][postidx] -= \
                    full_stim[preidx]*net.state.weights[preidx, postidx, time]

    # Have neurons declare 'interest to fire'.
    for neuron in range(len(net.state.NeurAccum[time])):
        if net.state.NeurAccum[time][neuron] > net.params.get('FIRETH', 0.8):
            # Register 'interest to fire'.
            wantToFire[neuron] = 1

    # STAGE II: Implement constraints from net-level considerations.
    # Example: WTA. No resitrictions from net level yet. All neurons that
    # want to fire will fire.
    net.state.firingCells = wantToFire

    # Barrel shift history
    net.state.fireHist[:-1, np.where(np.array(full_stim) != 0)[0]] = \
        net.state.fireHist[1:, np.where(np.array(full_stim) != 0)[0]]
    # Save last firing time for all cells that fired in this time step.
    net.state.fireHist[net.DEPTH, np.where(np.array(full_stim) != 0)[0]] = \
        time

    # Load 'NN'.
    net.state.fireCells[time] = full_stim


def plast(net, time):

    if time+2 > net.epochs:
        return

    rawin = net.rawin # Raw input
    stimin = net.stimin[:, time] # Stimulus input

    full_stim = np.bitwise_or([int(x) for x in rawin], [int(x) for x in stimin])

    net.state.weights[:, :, time+1] = net.state.weights[:, :, time]

    # For every neuron in the raw input
    for neuron in range(len(full_stim)):

        # If neuron is not set to fire (full_stim > 0) just skip the neuron
        if full_stim[neuron] == 0:
            continue

        # For every presynaptic input the neuron receives.
        for preidx in np.where(net.ConnMat[:, neuron, 0] != 0)[0]:
            w,b = net.ConnMat[preidx, neuron, 0:2]
            if (time - np.max(net.state.fireHist[:, preidx])) <= net.LTPWIN:
                # -FIX- parametrise learning step.
                # Actually, link it to bias for devices.
                p = 1.0/net.pulse(w, b, net.LTP_V, net.LTP_pw)
                if net.params.get('NORMALISE', False):
                    net.state.weights[preidx, neuron, time+1] = normalise_weight(net, p)
                else:
                    net.state.weights[preidx, neuron, time+1] = p
                net.log(" LTP --- spiking synapse %d -- %d" % (preidx, neuron))

        # For every postsynaptic input the neuron receives.
        for postidx in np.where(net.ConnMat[neuron, :, 0] != 0)[0]:
            w,b=net.ConnMat[neuron,postidx,0:2]
            if (time - np.max(net.state.fireHist[:, postidx])) <= net.LTDWIN:
                # -FIX- parametrise learning step.
                # Actually, link it to bias for devices.
                p = 1.0/net.pulse(w, b, net.LTD_V, net.LTD_pw)
                if net.params.get('NORMALISE', False):
                    net.state.weights[neuron, postidx, time+1] = normalise_weight(net, p)
                else:
                    net.state.weights[neuron, postidx, time+1] = p
                net.log(" LTD --- spiking synapse %d -- %d" % (neuron, postidx))

    # For every valid connection between neurons, find out which the
    # corresponding memristor is. Then, if the weight is still uninitialised
    # take a reading and ensure that the weight has a proper value.
    for preidx in range(len(rawin)):
        for postidx in range(len(rawin)):
            if net.ConnMat[preidx, postidx, 0] != 0:
                w, b = net.ConnMat[preidx, postidx, 0:2]
                if net.state.weights[preidx, postidx, time] == 0.0:
                    net.state.weights[preidx, postidx, time] = \
                        1.0/net.read(w, b, "NN")


def additional_data(net):
    # This function should return any additional data that might be produced
    # by this core. In this particular case there are None.
    return None
