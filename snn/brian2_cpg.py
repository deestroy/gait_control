from brian2 import *

class QuadrupedCPG:
    def __init__(self, dt=0.01):

        defaultclock.dt = dt * second
        # 4 oscillators for 4 legs: Front Left (FL), Front Right (FR), Rear Left (RL), Rear Right (RR)
        self.N = 4  # one oscillator per leg

        eqs = '''
        dv/dt = (I - v) / tau : 1
        I : 1
        tau : second
        '''

        self.neurons = NeuronGroup(self.N, eqs, threshold='v>1',
                                   reset='v=0', method='euler')

        self.neurons.v = 0
        self.neurons.I = 1.2
        self.neurons.tau = 0.05 * second

        # Mutual inhibition (left-right coupling)
        self.syn = Synapses(self.neurons, self.neurons,
                            on_pre='v_post -= 0.5')

        # Diagonal legs in phase
        # Indices: 0=FL, 1=FR, 2=RL, 3=RR
        self.syn.connect(i=[0,3,1,2], j=[1,2,0,3])

        self.spikemon = SpikeMonitor(self.neurons)

        self.net = Network(self.neurons, self.syn, self.spikemon)

    def step(self):
        self.net.run(defaultclock.dt)
        return self.spikemon.count
