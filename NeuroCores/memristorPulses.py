import numpy as np

from arc1pyqt.VirtualArC.parametric_device import ParametricDevice as Device

class memristorPulses:
    def __init__(self, dt, Ap, An, a0p, a1p, a0n, a1n, tp, tn, R): # initialise a device with parameters
        self.dt = dt
        self.Ap = Ap
        self.An = An
        self.a0p = a0p
        self.a1p = a1p
        self.a0n = a0n
        self.a1n = a1n
        self.tp = tp
        self.tn = tn
        self.R = R

    def ResistancePredict(self, pulseList):  # return the list of R for different pulses
        memristor = Device(self.Ap, self.An, self.a0p, self.a1p, self.a0n, self.a1n, self.tp, self.tn)
        res = []
        f = open("C:/Users/jh1d18/debug_log.txt", "a")
        for i in pulseList:
            memristor.initialise(self.R)
            f.write('pulsechoice')
            line = ' '.join(str(x) for x in i)
            f.write(line + ', ')
            for timestep in range(int(i[1]/self.dt)):
                memristor.step_dt(i[0], self.dt)
                #print('new R: %f, old R: %f, mag: %f, pw: %f' % (self.R, memristor.Rmem, i[0], i[1]))
            f.write('res:%f\n'% memristor.Rmem)
            res.append(memristor.Rmem)
        f.close()
        del memristor
        print('pulseList:', pulseList)
        print('res:', res)
        return res

    def BestPulseChoice(self, R_expect, pulseList):  # return the pulse that can make the device reach the expected resistance
        res = self.ResistancePredict(pulseList)
        res_dist = np.absolute(np.array(res) - R_expect)
        print('res dist:', res_dist)
        f = open("C:/Users/jh1d18/debug_log.txt", "a")
        f.write('res dist:')
        line = ', '.join(str(i) for i in list(res_dist))
        f.write(line + '\n')
        f.close()
        res_index = np.argmin(res_dist)

        return pulseList[res_index]
