import numpy as np
from .memristorPulses import memristorPulses as memristorPulses
# This core implements tempotron learning rule with temporal coding version LIF.
# This core can be only applied to two-layer NN (one input layer and one output layer)


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
    # Renormalise weights if needed

    for postidx in range(len(net.ConnMat)):
        # For every presynaptic input the neuron receives.
        for preidx in np.where(net.ConnMat[:, postidx, 0] != 0)[0]:
            w, b=net.ConnMat[preidx, postidx, 0:2]
            net.state.weights[preidx, postidx - net.inputNum, 0] = 1.0/net.read(w, b)
#            f = open("C:/Users/jh1d18/debug_log.txt", "a")
#            f.write('device intial state: %f, w: %d, b: %d\n' % (net.read(w, b), w, b))
#            f.close()
            if net.params.get('NORMALISE', False):
                old_weight = net.state.weights[preidx, postidx - net.inputNum, 0]
                new_weight = normalise_weight(net, old_weight)
                net.state.weights[preidx, postidx - net.inputNum, 0] = new_weight

def k(net, v0, t_diff):
    tau = net.params.get('TAU', 20e-3)
    tau_s = net.params.get('TAUS', 5e-3)
    k = v0 * (np.exp((-1) * t_diff / tau) - np.exp((-1) * t_diff / tau_s))
    return k

def t_i_hist(net, preidx, time_start, time): # return all firing time before current timestep
    stimulus = net.stimin[:, :].T # Stimulus input for all timestep
    inputStimMask = np.hstack((np.ones(net.inputNum), np.zeros(net.outputNum)))
    inputStimulus = stimulus * inputStimMask

    t_i = []
    for t in range(time_start, time+1):
        if inputStimulus[t, preidx]:
            t_i.append(t)

    return np.array(t_i)

def neurons(net, time, phase = 'training'):

    rawin = net.rawin # Raw input
    rawinPseudo = net.rawinPseudo # latest fire history without wta
    if phase == 'test':
        stimin = net.stiminForTesting[:, time] # Stimulus input for current timestep
    else:
        stimin = net.stimin[:, time] # input stimuli at this time step
    outputSpike = net.outputSpike # signal to indicate if the output spike is generated

    inputStimMask = np.hstack((np.ones(net.inputNum), np.zeros(net.NETSIZE - net.inputNum)))  # mask matrix to extract input spikes. Size: NETSIZE
    outputLabelMask = np.hstack((np.zeros(net.NETSIZE - net.outputNum), np.ones(net.outputNum))) # mask matrix to extract output labels.  Size: NETSIZE

    inputStim = np.bitwise_and([int(x) for x in inputStimMask], [int(x) for x in stimin])   # split input stimulus and output labels
    outputLabel = np.bitwise_and([int(x) for x in outputLabelMask], [int(x) for x in stimin])

    full_stim = np.bitwise_or([int(x) for x in rawin], [int(x) for x in inputStim])
    net.log("**** FULL_STIM = ", full_stim)

    tau = net.params.get('TAU', 20e-3)
    tau_s = net.params.get('TAUS', 5e-3)

    if time > 0:
        # if this isn't the first step copy the accumulators
        # from the previous step onto the new one
        if phase = 'test':
            net.state.NeurAccumForTest[time] = net.state.NeurAccumForTest[time-1]
        else:
            net.state.NeurAccum[time] = net.state.NeurAccum[time-1]

    wantToFire = len(net.ConnMat)*[0]

    # Gather/define other pertinent data to function of neuron.
    dt = net.params.get('TIMESTEP', 1e-3)
    # STAGE I: See what neurons do 'freely', i.e. without the constraints of
    # WTA or generally other neurons' activities.
    t_max = tau * tau_s * np.log(tau / tau_s) / (tau - tau_s)
    v_max = k(net, 1, t_max)
    v_0 = 1 / v_max
    v_rest = net.params.get('VREST', 0)
    for postidx in range(net.inputNum, net.NETSIZE):
        if phase = 'test':
            for preidx in np.where(net.ConnMat[:, postidx, 0] != 0)[0]:

                t_i = t_i_hist(net, preidx, net.state.lastSpikeTrain + 1, time)

                if t_i.size != 0:
                    t_diff = (time - t_i) * dt
                    K = k(net, v_0, t_diff)
                    input_contrib = sum(K)
                else:
                    input_contrib = 0

                # Excitatory case
                if net.ConnMat[preidx, postidx, 2] > 0:
                    # net.log("Excitatory at %d %d" % (preidx, postidx))
                    # Accumulator increases as per standard formula.
                    net.state.NeurAccumForTest[time][postidx - net.inputNum] += \
                        input_contrib * net.weightsForTest[preidx, postidx - net.inputNum, time]

                    net.log("POST=%d PRE=%d NeurAccum=%g input contribution=%g weight=%g" % \
                        (postidx, preidx, net.state.NeurAccumForTest[time][postidx - net.inputNum], \
                         input_contrib, net.weightsForTest[preidx, postidx - net.inputNum, time]))

                # Inhibitory case
                elif net.ConnMat[preidx, postidx, 2] < 0:
                    # Accumulator decreases as per standard formula.
                    net.state.NeurAccumForTest[time][postidx - net.inputNum] -= \
                        input_contrib * net.weightsForTest[preidx, postidx - net.inputNum, time]

            net.state.NeurAccumForTest[time][postidx] += v_rest
        else:
        #For every presynaptic input the neuron receives.
            for preidx in np.where(net.ConnMat[:, postidx, 0] != 0)[0]:

                t_i = t_i_hist(net, preidx, net.state.lastSpikeTrain + 1, time)

                if t_i.size != 0:
                    t_diff = (time - t_i) * dt
                    K = k(net, v_0, t_diff)
                    input_contrib = sum(K)
                else:
                    input_contrib = 0

                # Excitatory case
                if net.ConnMat[preidx, postidx, 2] > 0:
                    # net.log("Excitatory at %d %d" % (preidx, postidx))
                    # Accumulator increases as per standard formula.
                    net.state.NeurAccum[time][postidx - net.inputNum] += \
                        input_contrib * net.state.weights[preidx, postidx - net.inputNum, time]

                    net.log("POST=%d PRE=%d NeurAccum=%g input contribution=%g weight=%g" % \
                        (postidx, preidx, net.state.NeurAccum[time][postidx - net.inputNum], \
                         input_contrib, net.state.weights[preidx, postidx - net.inputNum, time]))

                # Inhibitory case
                elif net.ConnMat[preidx, postidx, 2] < 0:
                    # Accumulator decreases as per standard formula.
                    net.state.NeurAccum[time][postidx - net.inputNum] -= \
                        input_contrib*net.state.weights[preidx, postidx - net.inputNum, time]

            net.state.NeurAccum[time][postidx] += v_rest

    if phase == 'test':
        for neuron in range(len(net.state.NeurAccumForTest[time])):
            if net.state.NeurAccumForTest[time][neuron] > net.params.get('FIRETH', 0.8):
                # Register 'interest to fire'.
                wantToFire[neuron + net.inputNum] = 1
            if net.state.NeurAccumForTest[time][neuron] > net.state.voltMax[neuron]:
                net.state.voltMax[neuron] = net.state.NeurAccumForTest[time][neuron]
                net.state.tMax = time

        if sum(wantToFire) > 0:
            outputSpike = 1
        else:
            outputSpike = 0

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
        net.state.outputFlag = outputSpike
        # Load 'NN'.
        net.state.fireCellForTest[time] = full_stim
        net.state.spikeTrain_cnt += 1
        net.spikeTrainStep = net.state.spikeTrain_cnt

        if net.state.spikeTrain_cnt == net.state.spikeTrain:
            SumOfFireHistInOneTrain = np.sum(net.state.fireCells[time+1-net.state.spikeTrain : time+2], axis = 1)
            FireHistInOneTrain = np.where(SumFireHistInOneTrain > 0, 1, 0)
            net.state.errorList[(time+1) // net.state.spikeTrain] = FireHistInOneTrain - outputLable
    else:
        # Have neurons declare 'interest to fire'.
        for neuron in range(len(net.state.NeurAccum[time])):
            if net.state.NeurAccum[time][neuron] > net.params.get('FIRETH', 0.8):
                # Register 'interest to fire'.
                wantToFire[neuron] = 1
            if net.state.NeurAccum[time][neuron] > net.state.voltMax[neuron]:
                net.state.voltMax[neuron] = net.state.NeurAccum[time][neuron] #  size: netsize - inputNum
                net.state.tMax = time

        if sum(wantToFire) > 0:
            outputSpike = 1
        else:
            outputSpike = 0

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
        net.state.outputFlag = outputSpike
        # Load 'NN'.
        net.state.fireCells[time] = full_stim
        net.state.spikeTrain_cnt += 1
        net.spikeTrainStep = net.state.spikeTrain_cnt

        if net.state.spikeTrain_cnt == net.state.spikeTrain:
            SumOfFireHistInOneTrain = np.sum(net.state.fireCells[time+1-net.state.spikeTrain : time+2], axis = 1)
            FireHistInOneTrain = np.where(SumFireHistInOneTrain > 0, 1, 0)
            net.state.errorList[(time+1) // net.state.spikeTrain] = FireHistInOneTrain - outputLable

def plast(net, time):

    if time + 1 != net.epochs:
        net.state.weights[:, :, time+1] = net.state.weights[:, :, time]

    if net.state.spikeTrain_cnt != net.spikeTrain:
        return

    rawin = net.rawin # Raw input, the fire hist this time step
    stimin = net.stimin[:, time] # Stimulus input
    outputSpike = net.outputSpike # signal to indicate if the output spike is generated

    inputStimMask = np.hstack((np.ones(net.inputNum), np.zeros(net.NETSIZE - net.inputNum))) # mask matrix to extract input spikes. Size: NETSIZE
    outputLabelMask = np.hstack((np.zeros(net.NETSIZE - net.outputNum), np.ones(net.outputNum))) # mask matrix to extract output labels. Size: NETSIZE

    inputStim = np.bitwise_and([int(x) for x in inputStimMask], [int(x) for x in stimin])   # input spike matrix. Size: NETSIZE
    outputLabel = np.bitwise_and([int(x) for x in outputLabelMask], [int(x) for x in stimin]) # output label matrix. Size: NETSIZE

    full_stim = np.bitwise_or([int(x) for x in rawin], [int(x) for x in inputStim])

    tau = net.params.get('TAU', 20e-3)
    tau_s = net.params.get('TAUS', 5e-3)

    t_max = tau * tau_s * np.log(tau / tau_s) / (tau - tau_s)
    v_max = k(net, 1, t_max)
    v_0 = 1 / v_max
    learningRate = net.params.get('LEARNINGRATE', 0.01)
    dt = net.params.get('TIMESTEP', 1e-3)
    # For every neuron in the raw input
    for neuron in range(len(full_stim)):

        # do not update weights for input neurons, for the case that there is no output neuron fires, and for the case that correct neuron fires
        if neuron < net.inputNum or rawin[neuron] == outputLabel[neuron]:
            continue

        # For every presynaptic input the neuron receives.
        for preidx in np.where(net.ConnMat[:, neuron, 0] != 0)[0]:
            t_i = t_i_hist(net, preidx, net.state.lastSpikeTrain + 1, time)
            print("t_i", t_i)
            print("t_max", net.state.tMax)
            t_diff = (net.state.tMax - t_i) * dt
            print("t_diff", t_diff)
            t_diff = np.where(t_diff > 0, t_diff, 0)
            print("t_diff", t_diff)
            K = k(net, v_0, t_diff)
            print("K", K)
            input_contrib = sum(K)
            print("input_contrib", input_contrib)
            if rawin[neuron] == 1 and outputLabel[neuron] == 0: # if a neuron is fired when it is not expected to fire
                net.state.errorList[neuron + net.outputNum - fullNum, net.state.errorSteps_cnt] = 1
                dW = (-1) * learningRate * input_contrib
                pulseList = net.pos_pulseList                                                # weights need to be decreased so resistance should be increased
            elif rawin[neuron] == 0 and outputLabel[neuron] == 1:
                net.state.errorList[neuron + net.outputNum - fullNum, net.state.errorSteps_cnt] = -1
                dW = learningRate * input_contrib
                pulseList = net.neg_pulseList
            print("weight change:", dW)
            w,b = net.ConnMat[preidx, neuron, 0:2]
            R = net.read(w, b) # current R
            net.state.R[preidx, neuron - net.inputNum, time*4+1] = R
            if allNeuronsThatFire[preidx] == 0:
                continue
            grad = error[neuron] * allNeuronsThatFire[preidx]
            dW = (-1) * learningRate * grad
#            if dW > 0: # conductance needs to be larger, so a negative pulse is suplied
#                pulseList = net.neg_pulseList
#            else:
#                pulseList = net.pos_pulseList
            p = 1 / R
            if net.params.get('NORMALISE', False):
                p_norm = normalise_weight(net, p)
                p_expect = dW + p_norm
                R_expect = de_normalise_resistance(net, p_expect)
            else:
                p_expect = dW + p # new weights
                R_expect = 1 / p_expect #expected R
            # look up table mapping
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
                R = net.read(w, b)
            R_real = net.read(w, b)
            net.state.R[preidx, neuron - net.inputNum, time*4+2] = R_real
            p_real = 1 / R_real
            p_error = p_real - p_expect
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

    net.state.voltMax = np.array((net.NETSIZE - net.inputNum)*[0.0])
    net.state.tMax = 0
    net.state.spikeTrain_cnt = 0
    net.state.errorSteps_cnt += 1
    net.spikeTrainStep = net.state.spikeTrain_cnt
    net.state.lastSpikeTrain = time
    # For every valid connection between neurons, find out which the
    # corresponding memristor is. Then, if the weight is still uninitialised
    # take a reading and ensure that the weight has a proper value.
#    for preidx in range(len(rawin)):
#        for postidx in range(len(rawin)):
#            if net.ConnMat[preidx, postidx, 0] != 0:
#                w, b = net.ConnMat[preidx, postidx, 0:2]
#                if net.state.weights[preidx, postidx - net.inputNum, time] == 0.0:
#                    net.state.weights[preidx, postidx - net.inputNum, time] = \
#                        1.0/net.read(w, b, "NN")

def additional_data(net):
    # This function should return any additional data that might be produced
    # by this core. In this particular case there are None.
    return None
