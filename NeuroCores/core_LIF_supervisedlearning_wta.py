import numpy as np
from .memristorPulses import memristorPulses as memristorPulses
# This core implements feed forward NNs with LIF neurons. The output neurons of the NN can fire without restriction.
# Synapses are updated according to back propagation SGD, with the derivative of the step function replaced by noise.
# This core can be used for multi-layer cases.

def normalise_weight(net, w):
    PCEIL = 1.0/net.params['PFLOOR'] # conductance ceil
    PFLOOR = 1.0/net.params['PCEIL'] #  conductance floor

    val = net.params['WEIGHTSCALE']*(float(w) - PFLOOR)/(PCEIL - PFLOOR)

    # Clamp weights in-between 0.0 and 1.0
    if val < 0.0:
        return 0.0
    elif val > 1.0:
        return 1.0
    else:
        return val

def de_normalise_resistance(net, w):
    PCEIL = 1.0/net.params['PFLOOR'] # conductance ceil
    PFLOOR = 1.0/net.params['PCEIL'] #  conductance floor

    C = w * (PCEIL - PFLOOR) / net.params['WEIGHTSCALE'] + PFLOOR
    R = 1 / C
    return R

def init(net):
    # make sure all counters are reset
    net.spikeTrain_cnt = 0
    net.errorSteps_cnt = 0
    net.errorStepsForTest_cnt = 0

    for postidx in range(len(net.ConnMat)):
        # For every presynaptic input the neuron receives.
        for preidx in np.where(net.ConnMat[:, postidx, 0] != 0)[0]:
            w, b=net.ConnMat[preidx, postidx, 0:2]
            net.state.weights[preidx, postidx - net.inputNum, 0] = 1.0/net.read(w, b)
            f = open("C:/Users/jh1d18/debug_log.txt", "a")
            f.write('device intial state: %f, w: %d, b: %d\n' % (net.read(w, b), w, b))
            f.close()
            if net.params.get('NORMALISE', False):
                old_weight = net.state.weights[preidx, postidx - net.inputNum, 0]
                new_weight = normalise_weight(net, old_weight)
                net.state.weights[preidx, postidx - net.inputNum, 0] = new_weight

def softmax(x):
    # Compute softmax values for each sets of scores in x
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def neurons(net, time):

    rawin = net.rawin # Raw input
    stimin = net.stimin[:, time] # Stimulus input for current timestep

    inputStimMask = np.hstack((np.ones(net.inputNum), np.zeros(net.outputNum)))
    outputLabelMask = np.hstack((np.zeros(net.inputNum), np.ones(net.outputNum)))

    inputStim = np.bitwise_and([int(x) for x in inputStimMask], [int(x) for x in stimin])   # split input stimulus and output labels
    outputLabel = np.bitwise_and([int(x) for x in outputLabelMask], [int(x) for x in stimin])

    full_stim = np.bitwise_or([int(x) for x in rawin], [int(x) for x in inputStim])
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
                    full_stim[preidx] * net.state.weights[preidx, postidx - net.inputNum, time]

            # Inhibitory case
            elif net.ConnMat[preidx, postidx, 2] < 0:
                # Accumulator decreases as per standard formula.
                net.state.NeurAccum[time][postidx] -= \
                    full_stim[preidx]*net.state.weights[preidx, postidx - net.inputNum, time]

        net.log("POST=%d NeurAccum=%g in step %d" % (postidx, net.state.NeurAccum[time][postidx], time))
        f = open("C:/Users/jh1d18/debug_log.txt", "a")
        f.write('--------------\n')
        f.write("POST=%d NeurAccum=%g in step %d\n" % (postidx, net.state.NeurAccum[time][postidx], time))
        f.close()
    # Have neurons declare 'interest to fire'.
    for neuron in range(len(net.state.NeurAccum[time])):
        if net.state.NeurAccum[time][neuron] > net.params.get('FIRETH', 0.8):
            # Register 'interest to fire'.
            wantToFire[neuron] = 1

    neuronID = -1
    if sum(wantToFire[-net.outputNum:]) > 1:
        neuronID = np.argmax(net.state.NeurAccum[time])
        wantToFire = len(net.ConnMat)*[0]
        wantToFire[neuronID] = 1

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
    net.state.fireCells[time] = wantToFire
    net.state.errorList[time] = wantToFire - outputLabel
    print('winnner takes all results:', wantToFire)


def plast(net, time):

    if time+2 > net.epochs:
        return

    rawin = net.rawin # Raw input
    stimin = net.stimin[:, time] # Stimulus input

    inputStimMask = np.hstack((np.ones(net.inputNum), np.zeros(net.outputNum)))
    inputStim = np.bitwise_and([int(x) for x in inputStimMask], [int(x) for x in stimin])
    outputLabelMask = np.hstack((np.zeros(net.inputNum), np.ones(net.outputNum)))
    outputLabel = np.bitwise_and([int(x) for x in outputLabelMask], [int(x) for x in stimin])
    allNeuronsThatFire = np.bitwise_or([int(x) for x in inputStim], [int(x) for x in rawin])

    net.state.weights[:, :, time+1] = net.state.weights[:, :, time]
    fullNum = len(rawin)
    softmaxRes = softmax(np.multiply(net.state.NeurAccum[time], rawin)[-net.outputNum:])    # softmax result
    #softmaxRes = softmax(net.state.NeurAccum[time])    # softmax result
    net.log('softmaxRes:', softmaxRes)
    error = len(net.ConnMat)*[0]
    noiseScale = net.params.get('NOISESCALE', 1e-6)
    learningRate = net.params.get('LEARNINGRATE', 1e-6)
    # For every neuron in the raw input
    # store R values before updating weights
    for postidx in range(len(net.ConnMat)):
        # For every presynaptic input the neuron receives.
        for preidx in np.where(net.ConnMat[:, postidx, 0] != 0)[0]:
            w, b=net.ConnMat[preidx, postidx, 0:2]
            r_pre = net.read(w, b)
            net.state.R[preidx, postidx - net.inputNum, time*4] = r_pre
            net.state.R[preidx, postidx - net.inputNum, time*4+1] = r_pre
            net.state.R[preidx, postidx - net.inputNum, time*4+2] = r_pre

    for neuron in range(fullNum-1, -1, -1):  # update from the reversed order, in case the error hasn't been updated
        if neuron >= fullNum - net.outputNum:   # output neurons
            # delta = (S - y^hat)*(y + noise)
            # gradient = delta * weight
            net.log('neuron: %d, softmaxRes: %f, outputLabel: %d, fire hist: %d' %(neuron, softmaxRes[neuron-(fullNum - net.outputNum)], outputLabel[neuron], rawin[neuron]))
            f = open("C:/Users/jh1d18/debug_log.txt", "a")
            f.write('neuron: %d, softmaxRes: %f, outputLabel: %d, fire hist: %d\n' %(neuron, softmaxRes[neuron-(fullNum - net.outputNum)], outputLabel[neuron], rawin[neuron]))
            if rawin[neuron] == outputLabel[neuron]:
                continue
            delta = (softmaxRes[neuron-(fullNum - net.outputNum)] - outputLabel[neuron])
            net.log('delta:', delta)
            f.write('delta:%f\n' % delta)
            error[neuron] = delta * (outputLabel[neuron] + np.random.rand() * noiseScale)
            net.log('error:', error[neuron])
            f.write('error:%f\n'% error[neuron])
            f.close()
        elif neuron < fullNum - net.outputNum and neuron >= net.inputNum:   # hidden neurons
            for postidx in np.where(net.ConnMat[neuron,:, 0] != 0)[0]: # add up all error back propagated from the next layer
                error[neuron] += error[postidx] * net.state.weights[neuron, postidx - net.inputNum, time+1] * np.random.rand() * noiseScale
        else:   # input neuron
            continue

        if error[neuron] == 0.0:
            continue

        # For every presynaptic input the neuron receives, back propogate the error
        for preidx in np.where(net.ConnMat[:, neuron, 0] != 0)[0]:
            w,b = net.ConnMat[preidx, neuron, 0:2]
            R = net.read(w, b) # current R
            net.log('current R:', R)
            net.state.R[preidx, neuron - net.inputNum, time*4+1] = R
            if allNeuronsThatFire[preidx] == 0:
                continue
            grad = error[neuron] * allNeuronsThatFire[preidx]
            dW = (-1) * learningRate * grad
            net.log('dW:', dW)
            f = open("C:/Users/jh1d18/debug_log.txt", "a")
            f.write('dW:%f\n' % dW)
#            if dW > 0: # conductance needs to be larger, so a negative pulse is suplied
#                pulseList = net.neg_pulseList
#            else:
#                pulseList = net.pos_pulseList
            f.write('current R :%f\n' % R)
            p = 1 / R
            f.write('current weight stored:%f, %f\n'% (net.state.weights[preidx, neuron - net.inputNum, time], net.state.weights[preidx, neuron - net.inputNum, time+1]))
            if net.params.get('NORMALISE', False):
                p_norm = normalise_weight(net, p)
                net.log('current weight:', p_norm)
                f.write('current weight:%f\n'% p_norm)
                p_expect = dW + p_norm
                net.log("final weight after normalised should be: %f" % p_expect)
                f.write("final weight after normalised should be: %f\n" % p_expect)
                R_expect = de_normalise_resistance(net, p_expect)
            else:
                net.log('current weight:', p)
                f.write('current weight:%f\n'% p)
                p_expect = dW + p # new weights
                net.log("final weight should be: %f" % p_expect)
                f.write("final weight should be: %f\n" % p_expect)
                R_expect = 1 / p_expect #expected R
            # look up table mapping
            net.log('expected R:', R_expect)
            f.write('expected R:%f\n'% R_expect)
            f.close()
#            while abs(R - R_expect)/R_expect > 0.0035 and step <= 10:
            for step in range(net.maxSteps):
                if abs(R - R_expect)/R_expect < net.RTolerance:
                    break
                if R - R_expect > 0:    # resistance needs to be decreased
                    pulseList = net.neg_pulseList
                else:   # resistance needs to be increased
                    pulseList = net.pos_pulseList
                virtualMemristor = memristorPulses(net.dt, net.Ap, net.An, net.a0p, net.a1p, net.a0n, net.a1n, net.tp, net.tn, R)
                pulseParams = virtualMemristor.BestPulseChoice(R_expect, pulseList) # takes the best pulse choice
                del virtualMemristor
                net.log('pulse selected:', pulseParams)
                net.pulse(w, b, pulseParams[0], pulseParams[1])
                R = net.read(w, b)
                net.log('R:', R)
                f = open("C:/Users/jh1d18/debug_log.txt", "a")
                f.write('R:%f\n'% R)
                f.close()
            R_real = net.read(w, b)
            net.state.R[preidx, neuron - net.inputNum, time*4+2] = R_real
            net.log('new R:', R_real)
            f = open("C:/Users/jh1d18/debug_log.txt", "a")
            f.write('new R:%f\n'% R_real)
            p_real = 1 / R_real
            net.log('new weight:', p_real)
            f.write('new weight:%f\n'% p_real)
            p_error = p_real - p_expect
            net.log('weight error:', p_error)
            f.write('weight error:%f\n'% p_error)
            if net.params.get('NORMALISE', False):
                net.state.weights[preidx, neuron - net.inputNum - net.inputNum, time+1] = normalise_weight(net, p_real)
                net.state.weightsExpected[preidx, neuron - net.inputNum, time+1] = normalise_weight(net, p)
                net.state.weightsError[preidx, neuron - net.inputNum, time+1] = normalise_weight(net, p_error)
            else:
                net.state.weights[preidx, neuron - net.inputNum, time+1] = p_real
                net.state.weightsExpected[preidx, neuron - net.inputNum, time+1] = p
                net.state.weightsError[preidx, neuron - net.inputNum, time+1] = p_error
            net.log(" weight change for synapse %d -- %d from %f to %f in step %d" % (preidx, neuron, net.state.weights[preidx, neuron - net.inputNum, time], net.state.weights[preidx, neuron - net.inputNum, time+1], time))
            net.log('---------------')
            f.write(" weight change for synapse %d -- %d from %f to %f in step %d\n" % (preidx, neuron, net.state.weights[preidx, neuron - net.inputNum, time], net.state.weights[preidx, neuron - net.inputNum, time+1], time))
            f.write('---------------\n')
            f.close()

    for postidx in range(len(net.ConnMat)):
        # For every presynaptic input the neuron receives.
        for preidx in np.where(net.ConnMat[:, postidx, 0] != 0)[0]:
            w, b=net.ConnMat[preidx, postidx, 0:2]
            r_post = net.read(w, b)
            net.state.R[preidx, postidx - net.inputNum, time*4+3] = r_post
    # For every valid connection between neurons, find out which the
    # corresponding memristor is. Then, if the weight is still uninitialised
    # take a reading and ensure that the weight has a proper value.
#    for preidx in range(len(rawin)):
#        for postidx in range(len(rawin)):
#            if net.ConnMat[preidx, postidx, 0] != 0:
#                w, b = net.ConnMat[preidx, postidx, 0:2]
#                if net.state.weights[preidx, postidx - net.inputNum, time] == 0.0:
#                    net.state.weights[preidx, postidx - net.inputNum, time] = \
#                        1.0/net.read(w, b)

    net.state.errorSteps_cnt = time

def neuronsForTest(net, time):

    rawin = net.rawin # Raw input
    stimin = net.stiminForTesting[:, time] # Stimulus input for current timestep

    inputStimMask = np.hstack((np.ones(net.inputNum), np.zeros(net.outputNum)))
    outputLabelMask = np.hstack((np.zeros(net.inputNum), np.ones(net.outputNum)))

    inputStim = np.bitwise_and([int(x) for x in inputStimMask], [int(x) for x in stimin])   # split input stimulus and output labels
    outputLabel = np.bitwise_and([int(x) for x in outputLabelMask], [int(x) for x in stimin])

    full_stim = np.bitwise_or([int(x) for x in rawin], [int(x) for x in inputStim])
    net.log("**** FULL_STIM = ", full_stim)

    if time > 0:
        # if this isn't the first step copy the accumulators
        # from the previous step onto the new one
        net.state.NeurAccumForTest[time] = net.state.NeurAccumForTest[time-1]

    # reset the accumulators of neurons that have already fired
    for (idx, v) in enumerate(full_stim):
        if v != 0:
            net.state.NeurAccumForTest[time][idx] = 0.0

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

            # Inhibitory case
            elif net.ConnMat[preidx, postidx, 2] < 0:
                # Accumulator decreases as per standard formula.
                net.state.NeurAccumForTest[time][postidx] -= \
                    full_stim[preidx]*net.weightsForTest[preidx, postidx - net.inputNum]

        net.log("POST=%d NeurAccum=%g" % (postidx, net.state.NeurAccumForTest[time][postidx]))
        f = open("C:/Users/jh1d18/debug_log.txt", "a")
        f.write("POST=%d NeurAccum=%g\n" % (postidx, net.state.NeurAccumForTest[time][postidx]))
        f.close()
    # Have neurons declare 'interest to fire'.
    for neuron in range(len(net.state.NeurAccumForTest[time])):
        if net.state.NeurAccumForTest[time][neuron] > net.params.get('FIRETH', 0.8):
            # Register 'interest to fire'.
            wantToFire[neuron] = 1

    neuronID = -1
    if sum(wantToFire[-net.outputNum:]) > 1:
        neuronID = np.argmax(net.state.NeurAccumForTest[time])
        wantToFire = len(net.ConnMat)*[0]
        wantToFire[neuronID] = 1
    # STAGE II: Implement constraints from net-level considerations.
    # Example: WTA. No resitrictions from net level yet. All neurons that
    # want to fire will fire.
    net.state.firingCellsForTest = wantToFire

    # Barrel shift history
    net.state.fireHistForTest[:-1, np.where(np.array(full_stim) != 0)[0]] = \
        net.state.fireHistForTest[1:, np.where(np.array(full_stim) != 0)[0]]
    # Save last firing time for all cells that fired in this time step.
    net.state.fireHistForTest[net.DEPTH, np.where(np.array(full_stim) != 0)[0]] = \
        time

    # Load 'NN'.
    net.state.fireCellsForTest[time] = wantToFire
    net.state.errorListForTest[time] = wantToFire - outputLabel
    print('winnner takes all results:', wantToFire)

def additional_data(net):
    # This function should return any additional data that might be produced
    # by this core. In this particular case there are None.
    return None
