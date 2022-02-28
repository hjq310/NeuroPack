import numpy as np
from .memristorPulses import memristorPulses as memristorPulses
# This core implements Izhikevich neuron model

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
    if not net.params.get('NORMALISE', False):
        return

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

    c = net.params.get('C', -0.065) # after-spike reset value of v
    b = net.params.get('B', 0.2)    # sensitivity of u to v
    for postidx in range(len(net.ConnMat)):
        net.state.NeurAccum[0][postidx] = c
        net.state.NeurRecov[0][postidx] = b * net.state.NeurAccum[0][postidx]

def neurons(net, time, phase = 'training'):

    rawin = net.rawin # Raw input
    rawinPseudo = net.rawinPseudo # latest fire history without wta
    if phase == 'test':
        stimin = net.stiminForTesting[:, time] # Stimulus input for current timestep
    else:
        stimin = net.stimin[:, time] # input stimuli at this time step

    inputStimMask = np.hstack((np.ones(net.inputNum), np.zeros(net.NETSIZE - net.inputNum)))  # mask matrix to extract input spikes. Size: NETSIZE
    outputLabelMask = np.hstack((np.zeros(net.NETSIZE - net.outputNum), np.ones(net.outputNum))) # mask matrix to extract output labels.  Size: NETSIZE

    inputStim = np.bitwise_and([int(x) for x in inputStimMask], [int(x) for x in stimin])   # split input stimulus and output labels
    outputLabel = np.bitwise_and([int(x) for x in outputLabelMask], [int(x) for x in stimin]

    # Izhikevich neuron model: v' = 0.04v^2 + 5v + 140 - u + I
    # u' = a(bv - u)
    # if v > v_peak_boundary: v <- c, u <- u + d
    # v: membrane voltage; u: recovery variable
    a = net.params.get('A', 0.02)   # time constant of u
    b = net.params.get('B', 0.2)    # sensitivity of u to v
    c = net.params.get('C', -0.065) # after-spike reset value of v
    d = net.params.get('D', 0.2)    # increment of u after spike
    dt = net.params.get('TIMESTEP', 1e-3)
    v_peak_boundary = net.params.get('VPEAK', 0.03)

    rawinArray = np.array(rawin) # size: NETSIZE - inputNum
    wantToFire = len(net.ConnMat)*[0]
    full_stim = np.bitwise_or([int(x) for x in wantToFire], [int(x) for x in inputStim])
    # Gather/define other pertinent data to function of neuron.
    leakage = net.params.get('LEAKAGE', 1.0)

    if time > 0:
        if phase == 'test':
            # if this isn't the first step copy the accumulators
            # from the previous step onto the new one
            net.state.NeurAccumForTest[time] = net.state.NeurAccumForTest[time-1]  # size: NETSIZE - inputNum
            net.state.NeurRecovForTest[time] = net.state.NeurRecovForTest[time-1]
            # reset the accumulators of neurons that have already fired
            net.state.NeurAccumForTest[time] = net.state.NeurAccumForTest[time] * np.where(rawinArray[net.inputNum : ]  == 0, 1, 0) + np.where(rawinArray[net.inputNum : ]  == 0, 0, c)
            net.state.NeurRecovForTest[time] = net.state.NeurRecovForTest[time] + d * np.where(rawinArray[net.inputNum : ]  == 0, 1, 0)
        else:
            # if this isn't the first step copy the accumulators
            # from the previous step onto the new one
            net.state.NeurAccum[time] = net.state.NeurAccum[time-1]  # size: NETSIZE - inputNum
            net.state.NeurRecov[time] = net.state.NeurRecov[time-1]
            net.log('membrane from last time step:', net.state.NeurAccum[time])
            net.log('membrane recoery from last time step:', net.state.NeurRecov[time])
            # reset the accumulators of neurons that have already fired
            net.state.NeurAccum[time] = net.state.NeurAccum[time] * np.where(rawinArray[net.inputNum : ]  == 0, 1, 0) + np.where(rawinArray[net.inputNum : ]  == 0, 0, c)
            net.state.NeurRecov[time] = net.state.NeurRecov[time] + d * np.where(rawinArray[net.inputNum : ]  == 0, 1, 0)
            net.log('membrane after reset:', net.state.NeurAccum[time])
            net.log('membrane recoery after reset:', net.state.NeurRecov[time])


    # For this example we'll make Izhkevich neurons - if changing this file a back-up
    # is strongly recommended before proceeding.

    #  -FIX- implementing 'memory' between calls to this function.
    # NeurAccum = len(net.ConnMat)*[0] #Define neuron accumulators.
    # Neurons that unless otherwise dictated to by net or ext input will
    # fire.

    # STAGE I: See what neurons do 'freely', i.e. without the constraints of
    # WTA or generally other neurons' activities.
    for postidx in range(net.inputNum, net.NETSIZE):
        if phase == 'test':
            v = net.state.NeurAccumForTest[time][postidx]
            u = net.state.NeurRecovForTest[time][postidx]
            dv = v * v * 0.04 + v * 5 + 140 - u
            du = a * (b * v - u)
        #For every presynaptic input the neuron receives.
            for preidx in np.where(net.ConnMat[:, postidx, 0] != 0)[0]:
            # if it's in the test phase
                # Excitatory case
                if net.ConnMat[preidx, postidx, 2] > 0:
                    # net.log("Excitatory at %d %d" % (preidx, postidx))
                    # Accumulator increases as per standard formula.
                    dv += full_stim[preidx] * net.weightsForTest[preidx, postidx - net.inputNum, time]

                    net.log("POST=%d PRE=%d NeurAccum=%g full_stim=%g weight=%g" % \
                        (postidx, preidx, net.state.NeurAccumForTest[time][postidx], \
                         full_stim[preidx], net.weightsForTest[preidx, postidx - net.inputNum, time]))

                # Inhibitory case
                elif net.ConnMat[preidx, postidx, 2] < 0:
                    # Accumulator decreases as per standard formula.
                    dv -= full_stim[preidx]*net.weightsForTest[preidx, postidx - net.inputNum, time]

            net.state.NeurAccumForTest[time][postidx] += dv * dt
            net.state.NeurRecovForTest[time][postidx] += du * dt
            if net.state.NeurAccumForTest[time][postidx - net.inputNum] > v_peak_boundary:
                wantToFire[postidx] = 1

        else:
            v = net.state.NeurAccum[time][postidx]
            u = net.state.NeurRecov[time][postidx]
            dv = v * v * 0.04 + v * 5 + 140 - u
            du = a * (b * v - u)
            for preidx in np.where(net.ConnMat[:, postidx, 0] != 0)[0]:
                # Excitatory case
                if net.ConnMat[preidx, postidx, 2] > 0:
                    # net.log("Excitatory at %d %d" % (preidx, postidx))
                    # Accumulator increases as per standard formula.
                    dv += full_stim[preidx] * net.state.weights[preidx, postidx - net.inputNum, time]

                    net.log("POST=%d PRE=%d NeurAccum=%g full_stim=%g weight=%g" % \
                        (postidx, preidx, net.state.NeurAccum[time][postidx], \
                         full_stim[preidx], net.state.weights[preidx, postidx - net.inputNum, time]))

                # Inhibitory case
                elif net.ConnMat[preidx, postidx, 2] < 0:
                    # Accumulator decreases as per standard formula.
                    dv -= full_stim[preidx]*net.state.weights[preidx, postidx - net.inputNum, time]

            net.state.NeurAccum[time][postidx] += dv * dt
            net.state.NeurRecov[time][postidx] += du * dt
            if net.state.NeurAccum[time][postidx] > v_peak_boundary:
                wantToFire[postidx] = 1

        full_stim = np.bitwise_or([int(x) for x in wantToFire], [int(x) for x in inputStim])
    net.state.firingCellsPseudo = wantToFire
    # STAGE II: Implement constraints from net-level considerations.
    # Example: WTA. No resitrictions from net level yet. All neurons that
    # want to fire will fire.
    net.state.firingCells = wantToFire

    # Barrel shift history
    if phase == 'test':
        net.state.fireHistForTest[:-1, np.where(np.array(full_stim) != 0)[0]] = \
            net.state.fireHistForTest[1:, np.where(np.array(full_stim) != 0)[0]]
        # Save last firing time for all cells that fired in this time step.
        net.state.fireHistForTest[net.DEPTH, np.where(np.array(full_stim) != 0)[0]] = \
            time
        net.state.fireCellsForTest[time] = wantToFire
    else:
        net.state.fireHist[:-1, np.where(np.array(full_stim) != 0)[0]] = \
            net.state.fireHist[1:, np.where(np.array(full_stim) != 0)[0]]
        # Save last firing time for all cells that fired in this time step.
        net.state.fireHist[net.DEPTH, np.where(np.array(full_stim) != 0)[0]] = \
            time
        net.state.fireCells[time] = wantToFire

    # Load 'NN'.
    net.state.fireCells[time] = wantToFire
    net.state.errorList = wantToFire - outputLabel

#############!!!!!!!!!!No valid learning rule yet!!!!!!!!!!!!!!#######################

def plast(net, time):

    if time+2 > net.epochs:
        return

    rawin = net.rawin # Raw input
    stimin = net.stimin[:, time] # Stimulus input
    rawinPseudo = net.rawinPseudo

    full_stim = np.bitwise_or([int(x) for x in rawin], [int(x) for x in stimin])

    net.state.weights[:, :, time+1] = net.state.weights[:, :, time]

    noiseScale = net.params.get('NOISESCALE', 0.01)
    learningRate = net.params.get('LEARNINGRATE', 0.001)

    directRandomTargetProj = np.matmul(net.state.fixedRandomWeights[:, net.outputNum:].T, outputLabel[net.outputNum:])

    for neuron in range(len(rawin)-1, -1, -1):  # update from the reversed order, in case the error hasn't been updated
        if neuron >= fullNum - net.outputNum:   # output neurons
            # delta = (y - y_hat)*noise
            # gradient = delta * input
            delta = (rawin[neuron] - outputLabel[neuron])
            if abs(net.state.NeurAccum[time][neuron - net.inputNum] - net.params.get('FIRETH', 0.001)) > net.params.get('FIRETH', 0.001):
                error[neuron] = 0
            else:
                error[neuron] = delta * 0.5 / net.params.get('FIRETH', 0.001)
            print("neuron %d has delta %f and error %f" % (neuron, delta, error[neuron]))
        elif neuron < fullNum - net.outputNum and neuron >= net.inputNum:   # hidden neurons
            if abs(net.state.NeurAccum[time][neuron - net.inputNum] - net.params.get('FIRETH', 0.001)) > net.params.get('FIRETH', 0.001):
                error[neuron] = 0
            else:
                error[neuron] = directRandomTargetProj[neuron] * 0.5 * net.state.NeurAccum[time][neuron - net.inputNum] / net.params.get('FIRETH', 0.001)
        else:   # input neuron
            continue

        if error[neuron] == 0.0:
            continue

        # For every presynaptic input the neuron receives, back propogate the error
        for preidx in np.where(net.ConnMat[:, neuron, 0] != 0)[0]:
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

    # For every valid connection between neurons, find out which the
    # corresponding memristor is. Then, if the weight is still uninitialised
    # take a reading and ensure that the weight has a proper value.
#    for preidx in range(len(rawin)):
#        for postidx in range(len(rawin)):
#            if net.ConnMat[preidx, postidx, 0] != 0:
#                w, b = net.ConnMat[preidx, postidx, 0:2]
#                if net.state.weights[preidx, postid - net.inputNumx, time] == 0.0:
#                    net.state.weights[preidx, postidx - net.inputNum, time] = \
#                        1.0/net.read(w, b, "NN")

    net.state.errorSteps_cnt = time

def additional_data(net):
    # This function should return any additional data that might be produced
    # by this core. In this particular case there are None.
    return None
