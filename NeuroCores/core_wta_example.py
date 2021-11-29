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
    # make sure all counters are reset
    net.spikeTrain_cnt = 0
    net.errorSteps_cnt = 0
    net.errorStepsForTest_cnt = 0
    # Renormalise weights if needed
    if not net.params.get('NORMALISE', False):
        return

    for postidx in range(len(net.ConnMat)):
        # For every presynaptic input the neuron receives.
        for preidx in np.where(net.ConnMat[:, postidx, 0] != 0)[0]:
            old_weight = net.state.weights[preidx, postidx - net.inputNum, 0]
            new_weight = normalise_weight(net, old_weight)
            net.state.weights[preidx, postidx - net.inputNum, 0] = new_weight

def neurons(net, time):

    rawin = net.rawin # Raw input
    stimin = net.stimin[:, time] # Stimulus input for current timestep
    outputSpike = net.outputSpike # signal to indicate if the output spike is generated
    neuronLocked = net.neuronLocked # signal to indeicate if there is a neuron locked
    lockedNeuronID = net.lockedNeuronID # variable to record locked Neuron ID

    full_stim = np.bitwise_or([int(x) for x in rawin], [int(x) for x in stimin])
    net.log("**** FULL_STIM = ", full_stim)

    if time > 0:
        # if this isn't the first step copy the accumulators
        # from the previous step onto the new one
        net.state.NeurAccum[time] = net.state.NeurAccum[time-1]

    # reset the accumulators of neurons that have already fired
    if outputSpike == 1:
        net.state.NeurAccum[time][:] = 0.0

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
                    full_stim[preidx] * net.state.weights[preidx, postidx - net.inputNum, time]

                # net.log("POST=%d PRE=%d NeurAccum=%g full_stim=%g weight=%g" % \
                    # (postidx, preidx, net.state.NeurAccum[time][postidx], \
                     # full_stim[preidx], net.state.weights[preidx, postidx, time]))

            # Inhibitory case
            elif net.ConnMat[preidx, postidx, 2] < 0:
                # Accumulator decreases as per standard formula.
                net.state.NeurAccum[time][postidx] -= \
                    full_stim[preidx]*net.state.weights[preidx, postidx - net.inputNum, time]

        print('neuron %d membrane voltage %g at time step %d' % (postidx, net.state.NeurAccum[time][postidx] ,time))

    # Have neurons declare 'interest to fire'.
    for neuron in range(len(net.state.NeurAccum[time])):
        if net.state.NeurAccum[time][neuron] > net.params.get('FIRETH', 0.8):
            # Register 'interest to fire'.
            wantToFire[neuron] = 1

    neuronID = -1
    net.log("**** wantToFire = ", wantToFire)
    if neuronLocked == 0 and sum(wantToFire[-net.outputNum:]) > 1:   # there is more than one spike generated
            neuronID = np.argmax(net.state.NeurAccum[time])
            wantToFire = len(net.ConnMat)*[0]
            wantToFire[neuronID] = 1
            outputSpike = 1 # set the signal
            neuronLocked = 1 # set the signal
            lockedNeuronID = neuronID   #record the locked neuron id
    elif neuronLocked == 1 and wantToFire[lockedNeuronID] == 1:
            wantToFire = len(net.ConnMat)*[0]
            wantToFire[lockedNeuronID] = 1
            outputSpike = 1 # set the signal
    else:
        outputSpike = 0 # reset the signal

    # STAGE II: Implement constraints from net-level considerations.
    # Example: WTA. No resitrictions from net level yet. All neurons that
    # want to fire will fire.
    net.log("**** winner take all fire = ", wantToFire)
    net.log("**** output flag = ", outputSpike)

    net.state.firingCells = wantToFire

    # Barrel shift history
    net.state.fireHist[:-1, np.where(np.array(full_stim) != 0)[0]] = \
        net.state.fireHist[1:, np.where(np.array(full_stim) != 0)[0]]
    # Save last firing time for all cells that fired in this time step.
    net.state.fireHist[net.DEPTH, np.where(np.array(full_stim) != 0)[0]] = \
        time

    # Load 'NN'.
    net.state.fireCells[time] = wantToFire
    net.state.outputFlag = outputSpike
    net.state.neuronFixed = neuronLocked
    net.state.fixedNeuronID = lockedNeuronID

def plast(net, time):

    if time+2 > net.epochs:
        return

    rawin = net.rawin # Raw input
    stimin = net.stimin[:, time] # Stimulus input
    outputSpike = net.outputSpike # signal to indicate if the output spike is generated

    full_stim = np.bitwise_or([int(x) for x in rawin], [int(x) for x in stimin])

    net.state.weights[:, :, time+1] = net.state.weights[:, :, time]

    # For every neuron in the raw input
    for neuron in range(len(full_stim)):

        if outputSpike == 0:   # if no spike has been generated, do not update weights
            break

        # If neuron is not set to fire (full_stim > 0) just skip the neuron
        if neuron >= net.prefixSum_layers[-2]:
            # For every input the neuron receives.
            if full_stim[neuron] == 1:
                for idx in np.where(net.ConnMat[:, neuron, 0] != 0)[0]:
                    w,b = net.ConnMat[idx, neuron, 0:2]
                    #if (time - np.max(net.state.fireHist[:, preidx])) <= net.LTPWIN:
                    if full_stim[idx] == 0:
                        # -FIX- parametrise learning step.
                        # Actually, link it to bias for devices.
                        p = 1.0/net.pulse(w, b, net.pos_voltOfPulseList[0], net.pos_pulsewidthOfPulseList[0])   # for stdp, only one pulse choice is stored in the list
                        if net.params.get('NORMALISE', False):
                            net.state.weights[idx, neuron - net.inputNum, time+1] = normalise_weight(net, p)
                            net.state.weightsExpected[idx, neuron - net.inputNum, time+1] = normalise_weight(net, p)
                            net.state.weightsError[idx, neuron - net.inputNum, time+1] = 0.0
                        else:
                            net.state.weights[idx, neuron - net.inputNum, time+1] = p
                            net.state.weightsExpected[idx, neuron - net.inputNum, time+1] = p
                            net.state.weightsError[idx, neuron - net.inputNum, time+1] = 0
                        net.log(" LTP --- spiking synapse %d -- %d" % (idx, neuron))
                    else:
                        p = 1.0/net.pulse(w, b, net.neg_voltOfPulseList[0], net.neg_pulsewidthOfPulseList[0])
                        if net.params.get('NORMALISE', False):
                            net.state.weights[idx, neuron - net.inputNum, time+1] = normalise_weight(net, p)
                            net.state.weightsExpected[idx, neuron - net.inputNum, time+1] = normalise_weight(net, p)
                            net.state.weightsError[idx, neuron - net.inputNum, time+1] = 0.0
                        else:
                            net.state.weights[idx, neuron - net.inputNum, time+1] = p
                            net.state.weightsExpected[idx, neuron - net.inputNum, time+1] = p
                            net.state.weightsError[idx, neuron - net.inputNum, time+1] = 0
                        net.log(" LTD --- spiking synapse %d -- %d" % (idx, neuron))
#            else:
#                for idx in np.where(net.ConnMat[:, neuron, 0] != 0)[0]:
#                    w,b = net.ConnMat[idx, neuron, 0:2]
#                    #if (time - np.max(net.state.fireHist[:, preidx])) <= net.LTPWIN:
#                    # -FIX- parametrise learning step.
#                    # Actually, link it to bias for devices.
#                    p = 1.0/net.pulse(w, b, net.pos_voltOfPulseList[1], net.pos_pulsewidthOfPulseList[1])   # for stdp, only one pulse choice is stored in the list
#                    if net.params.get('NORMALISE', False):
#                        net.state.weights[idx, neuron, time+1] = normalise_weight(net, p)
#                        net.state.weightsExpected[idx, neuron, time+1] = normalise_weight(net, p)
#                        net.state.weightsError[idx, neuron, time+1] = 0.0
#                    else:
#                        net.state.weights[idx, neuron, time+1] = p
#                        net.state.weightsExpected[idx, neuron, time+1] = p
#                        net.state.weightsError[idx, neuron, time+1] = 0
#                    net.log(" LTP --- spiking synapse %d -- %d" % (idx, neuron))
        else:
            continue
    # For every valid connection between neurons, find out which the
    # corresponding memristor is. Then, if the weight is still uninitialised
    # take a reading and ensure that the weight has a proper value.
    for preidx in range(len(rawin)):
        for postidx in range(len(rawin)):
            if net.ConnMat[preidx, postidx, 0] != 0:
                w, b = net.ConnMat[preidx, postidx, 0:2]
                if net.state.weights[preidx, postidx - net.inputNum, time] == 0.0:
                    net.state.weights[preidx, postidx - net.inputNum, time] = \
                        1.0/net.read(w, b)

    net.state.errorSteps_cnt = time

def neuronsForTest(net, time):

    rawin = net.rawin # Raw input
    stimin = net.stiminForTesting[:, time] # Stimulus input for current timestep
    outputSpike = net.outputSpike # signal to indicate if the output spike is generated
    neuronLocked = net.neuronLocked # signal to indeicate if there is a neuron locked
    #lockedNeuronID = net.lockedNeuronID # ??

    full_stim = np.bitwise_or([int(x) for x in rawin], [int(x) for x in stimin])
    net.log("**** FULL_STIM = ", full_stim)

    if time > 0:
        # if this isn't the first step copy the accumulators
        # from the previous step onto the new one
        net.state.NeurAccumForTest[time] = net.state.NeurAccumForTest[time-1]

    # reset the accumulators of neurons that have already fired
    if outputSpike == 1:
        net.state.NeurAccumForTest[time][:] = 0.0

    # For this example we'll make I&F neurons - if changing this file a back-up
    # is strongly recommended before proceeding.

    #  -FIX- implementing 'memory' between calls to this function.
    # NeurAccum = len(net.ConnMat)*[0] #Define neuron accumulators.
    # Neurons that unless otherwise dictated to by net or ext input will
    # fire.
    # Gather/define other pertinent data to function of neuron.
    leakage = net.params.get('LEAKAGE', 1.0)
    bias = np.array(len(net.ConnMat)*[leakage]) #No active biases.

    # STAGE I: See what neurons do 'freely', i.e. without the constraints of
    # WTA or generally other neurons' activities.
    for postidx in range(len(net.state.NeurAccumForTest[time])):
        # Unconditionally add bias term
        net.state.NeurAccumForTest[time][postidx] += bias[postidx]
        if net.state.NeurAccumForTest[time][postidx] < 0.0:
            net.state.NeurAccumForTest[time][postidx] = 0.0

        #For every presynaptic input the neuron receives.
        for preidx in np.where(net.ConnMat[:, postidx, 0] != 0)[0]:

            # Excitatory case
            if net.ConnMat[preidx, postidx, 2] > 0:
                # net.log("Excitatory at %d %d" % (preidx, postidx))
                # Accumulator increases as per standard formula.
                net.state.NeurAccumForTest[time][postidx] += \
                    full_stim[preidx] * net.weightsForTest[preidx, postidx - net.inputNum]

                # net.log("POST=%d PRE=%d NeurAccum=%g full_stim=%g weight=%g" % \
                    # (postidx, preidx, net.state.NeurAccumForTest[time][postidx], \
                     # full_stim[preidx], net.weightsForTest[preidx, postidx]))

            # Inhibitory case
            elif net.ConnMat[preidx, postidx, 2] < 0:
                # Accumulator decreases as per standard formula.
                net.state.NeurAccumForTest[time][postidx] -= \
                    full_stim[preidx]*net.weightsForTest[preidx, postidx - net.inputNum]

        print('neuron %d membrane voltage %g at time step %d' % (postidx, net.state.NeurAccumForTest[time][postidx] ,time))


    neuronID = np.argmax(net.state.NeurAccumForTest[time])
    print('neuronID:', neuronID)
    wantToFire = len(net.ConnMat)*[0]
    wantToFire[neuronID] = 1

    # STAGE II: Implement constraints from net-level considerations.
    # Example: WTA. No resitrictions from net level yet. All neurons that
    # want to fire will fire.
    net.log("**** winner take all fire = ", wantToFire)
    net.log("**** output flag = ", outputSpike)

    net.state.firingCellsForTest = wantToFire

    # Barrel shift history
    net.state.fireHistForTest[:-1, np.where(np.array(full_stim) != 0)[0]] = \
        net.state.fireHistForTest[1:, np.where(np.array(full_stim) != 0)[0]]
    # Save last firing time for all cells that fired in this time step.
    net.state.fireHistForTest[net.DEPTH, np.where(np.array(full_stim) != 0)[0]] = \
        time

    # Load 'NN'.
    net.state.fireCellsForTest[time] = wantToFire
    # net.state.outputFlag = outputSpike
    # net.state.neuronFixed = neuronLocked
    # net.state.fixedNeuronID = lockedNeuronID

def additional_data(net):
    # This function should return any additional data that might be produced
    # by this core. In this particular case there are None.
    return None
