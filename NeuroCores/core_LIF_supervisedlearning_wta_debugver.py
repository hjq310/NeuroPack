import numpy as np
from .memristorPulses import memristorPulses as memristorPulses
# This core implements feed forward NNs with LIF neurons. The output neurons of the NN can fire without restriction.
# Synapses are updated according to back propagation SGD, with the derivative of the step function replaced by noise.
# This core can be used for multi-layer cases.

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
    for postidx in range(len(net.ConnMat)):
        # For every presynaptic input the neuron receives.
        for preidx in np.where(net.ConnMat[:, postidx, 0] != 0)[0]:
            w, b=net.ConnMat[preidx, postidx, 0:2]
            net.state.weights[preidx, postidx - net.inputNum, 0] = 1.0/net.read(w, b)
            if net.params.get('NORMALISE', False):
                old_weight = net.state.weights[preidx, postidx - net.inputNum, 0]
                new_weight = normalise_weight(net, old_weight)
                net.state.weights[preidx, postidx - net.inputNum, 0] = new_weight

def softmax(x):
    # Compute softmax values for each sets of scores in x
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def neurons(net, time, phase = 'training'):

    rawin = net.rawin # Raw input
    rawinPseudo = net.rawinPseudo # latest fire history without wta
    if phase == 'test':
        stimin = net.stiminForTesting[:, time] # Stimulus input for current timestep
    else:
        stimin = net.stimin[:, time] # input stimuli at this time stepp

    inputStimMask = np.hstack((np.ones(net.inputNum), np.zeros(net.NETSIZE - net.inputNum)))  # mask matrix to extract input spikes. Size: NETSIZE
    outputLabelMask = np.hstack((np.zeros(net.NETSIZE - net.outputNum), np.ones(net.outputNum))) # mask matrix to extract output labels.  Size: NETSIZE

    inputStim = np.bitwise_and([int(x) for x in inputStimMask], [int(x) for x in stimin])   # split input stimulus and output labels
    outputLabel = np.bitwise_and([int(x) for x in outputLabelMask], [int(x) for x in stimin])

    # For this example we'll make I&F neurons - if changing this file a back-up
    # is strongly recommended before proceeding.

    #  -FIX- implementing 'memory' between calls to this function.
    # NeurAccum = len(net.ConnMat)*[0] #Define neuron accumulators.
    # Neurons that unless otherwise dictated to by net or ext input will
    # fire.
    rawinArray = np.array(rawinPseudo) # size: NETSIZE - inputNum
    wantToFire = len(net.ConnMat)*[0]
    full_stim = np.bitwise_or([int(x) for x in wantToFire], [int(x) for x in inputStim])
    # Gather/define other pertinent data to function of neuron.
    leakage = net.params.get('LEAKAGE', 1.0)

    if time > 0:
        if phase == 'test':
            # if this isn't the first step copy the accumulators
            # from the previous step onto the new one
            net.state.NeurAccumForTest[time] = net.state.NeurAccumForTest[time-1]  # size: NETSIZE - inputNum
            # reset the accumulators of neurons that have already fired
            net.state.NeurAccumForTest[time] = net.state.NeurAccumForTest[time] * np.where(rawinArray[net.inputNum : ]  == 0, 1, 0)
            # calculate the leakage term
            net.state.NeurAccumForTest[time] *= (1 - leakage)
        else:
            # if this isn't the first step copy the accumulators
            # from the previous step onto the new one
            net.state.NeurAccum[time] = net.state.NeurAccum[time-1]  # size: NETSIZE - inputNum
            net.log('membrane from last time step:', net.state.NeurAccum[time])
            # reset the accumulators of neurons that have already fired
            net.state.NeurAccum[time] = net.state.NeurAccum[time] * np.where(rawinArray[net.inputNum : ] == 0, 1, 0)
            net.log('membrane after reset:', net.state.NeurAccum[time])
            # calculate the leakage term
            net.state.NeurAccum[time] *= (1 - leakage)
            net.log('membrane after adding leakage:', net.state.NeurAccum[time])

    # STAGE I: See what neurons do 'freely', i.e. without the constraints of
    # WTA or generally other neurons' activities.
    for postidx in range(net.inputNum, net.NETSIZE):
        #For every presynaptic input the neuron receives.
        for preidx in np.where(net.ConnMat[:, postidx, 0] != 0)[0]:
            # if it's in the test phase
            if phase == 'test':
                # Excitatory case
                if net.ConnMat[preidx, postidx, 2] > 0:
                    # net.log("Excitatory at %d %d" % (preidx, postidx))
                    # Accumulator increases as per standard formula.
                    net.state.NeurAccumForTest[time][postidx - net.inputNum] += \
                        full_stim[preidx] * net.weightsForTest[preidx, postidx - net.inputNum]

                # Inhibitory case
                elif net.ConnMat[preidx, postidx, 2] < 0:
                    # Accumulator decreases as per standard formula.
                    net.state.NeurAccumForTest[time][postidx - net.inputNum] -= \
                        full_stim[preidx]*net.weightsForTest[preidx, postidx - net.inputNum]

            else:
                # Excitatory case
                if net.ConnMat[preidx, postidx, 2] > 0:
                    # net.log("Excitatory at %d %d" % (preidx, postidx))
                    # Accumulator increases as per standard formula.
                    net.state.NeurAccum[time][postidx - net.inputNum] += \
                        full_stim[preidx] * net.state.weights[preidx, postidx - net.inputNum, time]

                # Inhibitory case
                elif net.ConnMat[preidx, postidx, 2] < 0:
                    # Accumulator decreases as per standard formula.
                    net.state.NeurAccum[time][postidx - net.inputNum] -= \
                        full_stim[preidx]*net.state.weights[preidx, postidx - net.inputNum, time]

        if phase == 'test' and net.state.NeurAccumForTest[time][postidx - net.inputNum] > net.params.get('FIRETH', 0.001):
            wantToFire[postidx] = 1    # update the firehist to feedforward the spike
        elif phase == 'training' and net.state.NeurAccum[time][postidx - net.inputNum] > net.params.get('FIRETH', 0.001):
            wantToFire[postidx] = 1
        full_stim = np.bitwise_or([int(x) for x in wantToFire], [int(x) for x in inputStim])
#        net.log('updated full_stim:',full_stim)
        net.log("POST=%d NeurAccum=%g in step %d" % (postidx, net.state.NeurAccum[time][postidx - net.inputNum], time))
#        f = open("C:/Users/jh1d18/debug_log_withoutMems.txt", "a")
#        f.write('--------------\n')
#        f.write("POST=%d NeurAccum=%g in step %d\n" % (postidx, net.state.NeurAccum[time][postidx - net.inputNum], time))
#        f.close()
    net.log('free to fire neurons:', wantToFire)
    net.state.firingCellsPseudo = wantToFire # fire hist without wta. This info will be used to reset data.

    # STAGE II: Implement constraints from net-level considerations.
    # Example: WTA. No resitrictions from net level yet. All neurons that
    # want to fire will fire.
    # In this case, we implement winner-take-all to only allow the neuron with highest memrabne voltage to fire
    neuronID = -1
    if phase == 'test':
        if sum(wantToFire[-net.outputNum:]) > 1:
            neuronID = np.argmax(net.state.NeurAccumForTest[time][-net.outputNum:])
            winnerTakeAll = net.outputNum*[0]
            winnerTakeAll[neuronID] = 1
            wantToFire = wantToFire[ : net.NETSIZE - net.outputNum]
            wantToFire.extend(winnerTakeAll)
        # Barrel shift history
        net.state.fireHistForTest[:-1, np.where(np.array(full_stim) != 0)[0]] = \
            net.state.fireHistForTest[1:, np.where(np.array(full_stim) != 0)[0]]
        # Save last firing time for all cells that fired in this time step.
        net.state.fireHistForTest[net.DEPTH, np.where(np.array(full_stim) != 0)[0]] = \
            time
        # Load 'NN'.
        net.state.fireCellsForTest[time] = wantToFire
    else:
        if sum(wantToFire[-net.outputNum:]) > 1:
            neuronID = np.argmax(net.state.NeurAccum[time][-net.outputNum:])
            net.log('firing neuron id:', neuronID + net.NETSIZE - net.outputNum)
            winnerTakeAll = net.outputNum*[0]
            winnerTakeAll[neuronID] = 1
            wantToFire = wantToFire[ : net.NETSIZE - net.outputNum]
            wantToFire.extend(winnerTakeAll)
        net.log('wta fire state:', wantToFire)
        f = open("C:/Users/jh1d18/debug_log_withoutMems.txt", "a")
        for item in wantToFire[-net.outputNum : ]:
            f.write("%s" % item)
        f.write('---------------\n')
        f.close()
        # Barrel shift history
        net.state.fireHist[:-1, np.where(np.array(full_stim) != 0)[0]] = \
            net.state.fireHist[1:, np.where(np.array(full_stim) != 0)[0]]
        # Save last firing time for all cells that fired in this time step.
        net.state.fireHist[net.DEPTH, np.where(np.array(full_stim) != 0)[0]] = \
            time
        # Load 'NN'.
        net.state.fireCells[time] = wantToFire

    net.state.firingCells = wantToFire
    net.state.errorList[time] = wantToFire - outputLabel

def plast(net, time):

    if time+2 > net.epochs:
        return

    rawin = net.rawin # Raw input
    stimin = net.stimin[:, time] # Stimulus input
    rawinPseudo = net.rawinPseudo

    inputStimMask = np.hstack((np.ones(net.inputNum), np.zeros(net.NETSIZE - net.inputNum))) # mask matrix to extract input spikes. Size: NETSIZE
    outputLabelMask = np.hstack((np.zeros(net.NETSIZE - net.outputNum), np.ones(net.outputNum))) # mask matrix to extract output labels. Size: NETSIZE

    inputStim = np.bitwise_and([int(x) for x in inputStimMask], [int(x) for x in stimin])   # input spike matrix. Size: NETSIZE
    outputLabel = np.bitwise_and([int(x) for x in outputLabelMask], [int(x) for x in stimin]) # output label matrix. Size: NETSIZE

    allNeuronsThatFire = np.bitwise_or([int(x) for x in inputStim], [int(x) for x in rawinPseudo])

    net.state.weights[:, :, time+1] = net.state.weights[:, :, time]
    fullNum = len(rawin)
    softmaxRes = softmax(np.multiply(net.state.NeurAccum[time][-net.outputNum:], rawinPseudo[-net.outputNum:]))
    #softmaxRes = softmax(np.multiply(net.state.NeurAccum[time][-net.outputNum:], rawin[-net.outputNum:]))    # softmax result
    #softmaxRes = softmax(net.state.NeurAccum[time][-net.outputNum:])    # softmax result
    error = len(net.ConnMat)*[0]
    noiseScale = net.params.get('NOISESCALE', 1e-6)
    learningRate = net.params.get('LEARNINGRATE', 1e-6)
    # For every neuron in the raw input
    for neuron in range(fullNum-1, -1, -1):  # update from the reversed order, in case the error hasn't been updated
        if neuron >= fullNum - net.outputNum:   # output neurons
            # delta = (S - y^hat)*(y + noise)
            # gradient = delta * weight
            net.log('neuron: %d, softmaxRes: %f, outputLabel: %d, fire hist: %d' %(neuron, softmaxRes[neuron-(fullNum - net.outputNum)], outputLabel[neuron], rawin[neuron]))
            f = open("C:/Users/jh1d18/debug_log_withoutMems.txt", "a")
            f.write('neuron: %d, membrane: %f, softmaxRes: %f, outputLabel: %d, fire hist: %d, fire hist after wta:%d\n' %(neuron, net.state.NeurAccum[time][neuron - net.inputNum], softmaxRes[neuron-(fullNum - net.outputNum)], outputLabel[neuron], rawinPseudo[neuron], rawin[neuron]))
            if rawin[neuron] == outputLabel[neuron]:
                continue
            delta = (softmaxRes[neuron-(fullNum - net.outputNum)] - outputLabel[neuron])
            net.log('delta:', delta)
            f.write('delta:%f\n' % delta)
            #error[neuron] = delta * (1 + np.random.rand() * noiseScale)
            #error[neuron] = delta # using softamx(Vt) as the output for the output layer
            if abs(net.state.NeurAccum[time][neuron - net.inputNum] - net.params.get('FIRETH', 0.001)) > net.params.get('FIRETH', 0.001):
                error[neuron] = delta * (rawinPseudo[neuron])
            else:
                error[neuron] = delta * (rawinPseudo[neuron] + 0.5 * net.state.NeurAccum[time][neuron - net.inputNum]) / net.params.get('FIRETH', 0.001)
            net.log('error:', error[neuron])
            f.write('error:%f\n'% error[neuron])
            f.close()
        elif neuron < fullNum - net.outputNum and neuron >= net.inputNum:   # hidden neurons
            if abs(net.state.NeurAccum[time][neuron - net.inputNum] - net.params.get('FIRETH', 0.001)) > net.params.get('FIRETH', 0.001):
                sur_deriv = 0
            else:
                sur_deriv = 0.5 * net.state.NeurAccum[time][neuron - net.inputNum] / net.params.get('FIRETH', 0.001)
            for postidx in np.where(net.ConnMat[neuron,:, 0] != 0)[0]: # add up all error back propagated from the next layer
                delta_error = error[postidx] * net.state.weights[neuron, postidx - net.inputNum, time] * sur_deriv
                error[neuron] += delta_error
                net.log('error accumalted from neuron %d to neuron %d by %g to %g, orignal error is %g with a scale factor of %g' % (postidx, neuron, delta_error, error[neuron], error[postidx], net.state.weights[neuron, postidx - net.inputNum, time]))
            net.log('error, membrane vltage, and surrogate derivative for neuron %d :%g, %g, %g' % (neuron, net.state.NeurAccum[time][neuron - net.inputNum], sur_deriv, error[neuron]))
        else:   # input neuron
            continue

        if error[neuron] == 0.0:
            continue

        # For every presynaptic input the neuron receives, back propogate the error
        for preidx in np.where(net.ConnMat[:, neuron, 0] != 0)[0]:
            w,b = net.ConnMat[preidx, neuron, 0:2]
            if allNeuronsThatFire[preidx] == 0:
                continue
            grad = error[neuron] * allNeuronsThatFire[preidx]
            dW = (-1) * learningRate * grad
            net.log('dW:', dW)
            f = open("C:/Users/jh1d18/debug_log_withoutMems.txt", "a")
            f.write('dW:%g\n:' % dW)
            f.close()
#           new line for debuging
            p = dW + net.state.weights[preidx, neuron - net.inputNum, time+1]
#            if dW > 0: # conductance needs to be larger, so a negative pulse is suplied
#                pulseList = net.neg_pulseList
#            else:
#                pulseList = net.pos_pulseList
#            R = net.read(w, b) # current R
#            print('current R:', R)
#            p = dW + 1 / R # new weights
#            print(" final weight should be: %f" % p)
#            # look up table mapping
#            R_expect = 1 / p #expected R
#            print('expected R:', R_expect)
#            virtualMemristor = memristorPulses(net.dt, net.Ap, net.An, net.a0p, net.a1p, net.a0n, net.a1n, net.tp, net.tn, R)
#            pulseParams = virtualMemristor.BestPulseChoice(R_expect, pulseList) # takes the best pulse choice
#            print('pulse selected:', pulseParams)
#            net.pulse(w, b, pulseParams[0], pulseParams[1])
#            R_real = net.read(w, b)
#            print('new R:', R_real)
#            p_real = 1 / R_real
#            print('new weight:', p_real)
#            p_error = p_real - p
#            print('weight error:', p_error)
            if net.params.get('NORMALISE', False):
                net.state.weights[preidx, neuron - net.inputNum, time+1] = normalise_weight(net, p)
                #net.state.weightsExpected[preidx, neuron, time+1] = normalise_weight(net, p)
                #net.state.weightsError[preidx, neuron, time+1] = normalise_weight(net, p_error)
            else:
                net.state.weights[preidx, neuron - net.inputNum, time+1] = p
                #net.state.weightsExpected[preidx, neuron, time+1] = p
                #net.state.weightsError[preidx, neuron, time+1] = p_error
            net.log(" weight change for synapse %d -- %d from %g to %g in step %d" % (preidx, neuron, net.state.weights[preidx, neuron - net.inputNum, time], net.state.weights[preidx, neuron - net.inputNum, time+1], time))
            net.log('---------------')
            f = open("C:/Users/jh1d18/debug_log_withoutMems.txt", "a")
            f.write(" weight change for synapse %d -- %d from %g to %g in step %d\n" % (preidx, neuron, net.state.weights[preidx, neuron - net.inputNum, time], net.state.weights[preidx, neuron - net.inputNum, time+1], time))
            f.write('---------------\n')
            f.close()
    # For every valid connection between neurons, find out which the
    # corresponding memristor is. Then, if the weight is still uninitialised
    # take a reading and ensure that the weight has a proper value.
#    for preidx in range(len(rawin)):
#        for postidx in range(len(rawin)):
#            if net.ConnMat[preidx, postidx, 0] != 0:
#                w, b = net.ConnMat[preidx, postidx, 0:2]
#                if net.state.weights[preidx, postidx, time] == 0.0:
#                    net.state.weights[preidx, postidx, time] = \
#                        1.0/net.read(w, b, "NN")

    net.state.errorSteps_cnt = time

def additional_data(net):
    # This function should return any additional data that might be produced
    # by this core. In this particular case there are None.
    return None
