import numpy as np
import random
import tqdm as tqdm
import os 


# Helper functions
#  problem: t=t0 is a special case when hs_pos and hs_neg both are 1, so the function is not continuous !!!!
def doubgauss(t, t0, T1, T2, amp):
    '''
    Double gaussian function. What is that? Its a two gauss functions, where in case of t<t0 the first one is used, and in case of t>t0 the second one is used.

    t: time
    t0: time of appearance
    T1: rise time (decay of the left side t<t0)
    T2: decay time (decay of the right side t>t0)
    amp: amplitude
    '''

    dt = t - t0

    hs_neg = np.heaviside(-dt, 0)
    hs_pos = np.heaviside(dt, 0)

    return amp * (np.exp(-(dt**2) / (2*T1**2)) * hs_neg +
                  np.exp(-(dt**2) / (2*T2**2)) * hs_pos)


def write_to_file(file_pointer, list_of_variables):
    """ Write list of variables to file. 
    file_pointer: file pointer
    list_of_variables: list of variables to write to file
    """
    delimiter = ","
    file_pointer.write(delimiter.join(str(x)
                       for x in list_of_variables) + "\n")
                       


class Network:
    def __init__(self, tstop=28, stepp=0.005 * 1.e-3, sr=0.001):
        # set file pointers
        # this exact files location (absolut path)
        self_loc = os.path.dirname(os.path.abspath(__file__))
        savedir = os.path.join(self_loc, "results")
        self.rasterexc, self.rasterinh, self.timecourse, self.conn_f = self.init_file_pointers(savedir)

        # load parameters
        self.params = self.load_params()

        # set time variables
        self.tstop = tstop  # end of simulation (seconds),
        self.stepp = stepp  # until 0.002  - timestep in the simulation
        self.sr = sr  # sampling rate (seconds)

        # Init network
        self.gij_connections = self.init_connections(
            self.params.dim, self.params.pconn)  # connection matrix of 0s and 1s
        # weight matrix, randomly sampled weights from a lognaormal distribution (mu=muw, sigma=sigmaw)
        self.gij_weights = self.init_weights(
            self.params.dim, self.params.muw, self.params.sigmaw)
        self.gij_weights = self.limit_excitatory_weights(
            self.gij_weights, self.params.dimmexc, self.params.wmin, self.params.wmax)  # why only excitatory?!

        # init membrane pot. 'v'
        self.v = (-80 + 10 * (0.5 - np.random.rand(1000))) * 1.e-3

        # init refractory period??
        self.ww = np.zeros(self.params.dimmexc)  # ??
        self.tolsp = np.zeros(self.params.dim)  # ??
        # last spike time for neuron i,  10000000 in unknown (should be inf?)
        self.spike_flag = np.ones(self.params.dim) * np.inf

        # init conductances (G) and synaptic currents? (Q), eligibility trace (eligibility) and last update time (lastupdate), gijint, p_gabaB, q_gabaB
        self.GE_gabaa = np.zeros(self.params.dimmexc)
        self.GI_gabaa = np.zeros(self.params.diminh)
        self.GE_ampa = np.zeros(self.params.dimmexc)
        self.GI_ampa = np.zeros(self.params.diminh)
        self.GE_nmda = np.zeros(self.params.dimmexc)
        self.GI_nmda = np.zeros(self.params.diminh)
        self.GE_gabab = np.zeros(self.params.dimmexc)
        self.GI_gabab = np.zeros(self.params.diminh)

        self.QE_gabab = np.zeros(self.params.dimmexc)
        self.QI_gabab = np.zeros(self.params.diminh)

        # that, i think is the STDP curve
        self.eligibility = np.zeros(
            (self.params.dim, self.params.dim), dtype=float)
        # last update time for each connection
        self.lastupdate = np.zeros(
            (self.params.dim, self.params.dim), dtype=float)

        self.gijintEE = np.zeros(
            (self.params.dimmexc, self.params.dimmexc), dtype=int)
        self.gijintEI = np.zeros(
            (self.params.dimmexc, self.params.diminh), dtype=int)
        self.gijintIE = np.zeros(
            (self.params.diminh, self.params.dimmexc), dtype=int)
        self.gijintII = np.zeros(
            (self.params.diminh, self.params.diminh), dtype=int)

        self.p_gabaB = np.zeros(self.params.diminh)
        self.q_gabaB = np.zeros(self.params.diminh)

        self.rexc = 0
        self.rinh = 0

    def sample_population_activity(self, timet, dopamine):
        """ Sample population activity when its time, by writing values to the file 'timecourse'.
        Resets rexc and rinh to 0.
         """

        # printf("%lf\n",timet)
        list_of_variables = [timet, 
                             (self.rexc / self.params.dimmexc) / self.sr,
                             (self.rinh / self.params.diminh) / self.sr, 
                             (self.v[0] - self.params.EgabaB) * self.params.g_gabaB * self.GE_gabab[0], 
                             (self.v[0] - self.params.Ei) * self.GE_gabaa[0],
                             -(self.v[0] - self.params.Ee) * 4 * self.params.g_nmda * self.x_nmda(self.v[0]) * self.GE_nmda[0],
                             -(self.v[0] - self.params.Ee) * 4*self.GE_ampa[0],
                             self.gij_weights[0][150], self.eligibility[0][150], dopamine]
        write_to_file(self.timecourse, list_of_variables)

        self.rexc = 0  # maybe its the number of spikes in the last sampling period?
        self.rinh = 0  # maybe its the number of spikes in the last sampling period?

    def update_biophysical_properties(self, timet):
        ''' Updates G (conductances) and Q (synaptic currents?), compute v (membrane pot.), 
        get ww (?) (for excitatory), and update weights (using STDP).
        '''
        # excitatory units: GE, QI, v, ww
        for i in range(self.params.dimmexc):
            # input rate based conductnce update?
            if random.random() < (self.stepp * self.params.pconn * self.params.dimmexc * self.params.inputrate_on_exc):
                self.GE_ampa[i] += self.params.g_ampa
            if random.random() < (self.stepp * self.params.pconn * self.params.diminh * self.params.inputrate_on_inh):  #  isnt it dimmexc?
                self.GE_gabaa[i] += self.params.g_gabaa
            # external input based conductance update?
            if random.random() < (self.stepp * self.params.pconn * self.params.dimmexc * self.params.input_extern):
                self.GE_ampa[i] += self.params.g_ampa
            if random.random() < (self.stepp * self.params.pconn * self.params.dimmexc * self.params.input_extern_inh):
                self.GE_gabaa[i] += self.params.g_gabaa

            # update membrane potential
            self.v[i] += (self.params.glRS * (self.params.ElRS-self.v[i]) / self.params.CmRS +
                          self.params.glRS * self.params.deltaRS * np.exp(((self.v[i]-self.params.thr)/(self.params.deltaRS))) / self.params.CmRS +
                          self.GE_ampa[i] * (self.params.Ee-self.v[i]) / self.params.CmRS+self.GE_gabaa[i] * (self.params.Ei-self.v[i]) / self.params.CmRS +
                          self.x_nmda(self.v[i]) * self.params.g_nmda * self.GE_nmda[i] * (self.params.Ee-self.v[i]) / self.params.CmRS-self.ww[i] / self.params.CmRS +
                          self.params.g_gabaB *
                          self.GE_gabab[i] * (self.params.EgabaB -
                                              self.v[i]) / self.params.CmRS
                          ) * self.stepp * np.heaviside(timet-self.tolsp[i]-self.params.Trefr, 0)
            # update ??
            self.ww[i] += (-self.ww[i] + self.params.aRS * (self.v[i] -
                           self.params.ElRS)) * self.stepp / self.params.tauw

            # update conductances
            self.GE_ampa[i] += self.stepp * (-self.GE_ampa[i] / self.params.tau_decay_ampa)
            self.GE_nmda[i] += self.stepp * (-self.GE_nmda[i] / self.params.tau_decay_nmda)
            self.GE_gabaa[i] += self.stepp * (-self.GE_gabaa[i] / self.params.tau_decay_gabaa)

            self.GE_gabab[i] += self.stepp * (-self.GE_gabab[i] / self.params.tau_decay_gabaB +
                                              self.params.alpha_gabaB * self.QE_gabab[i])
            self.QE_gabab[i] += self.stepp * (-self.QE_gabab[i] / self.params.tau_rise_gabaB)

        # inhibitory units: GI, QI, v
        for i in range(self.params.dimmexc, self.params.dim):
            if random.random() < (self.stepp * self.params.inputrate_on_exc * self.params.pconn * self.params.dimmexc):
                self.GI_ampa[i-self.params.dimmexc] += self.params.g_ampa
            if random.random() < (self.stepp * self.params.inputrate_on_inh * self.params.pconn * self.params.diminh):
                self.GI_gabaa[i-self.params.dimmexc] += self.params.g_gabaa

            # update membrane potential
            self.v[i] += (self.params.glFS * (self.params.ElFS-self.v[i]) / self.params.CmFS +
                          self.params.glFS * self.params.deltaFS * np.exp(((self.v[i] - self.params.thr) / (self.params.deltaFS))) / self.params.CmFS +
                          self.GI_ampa[i-self.params.dimmexc] * (self.params.Ee - self.v[i]) / self.params.CmFS +
                          self.GI_gabaa[i-self.params.dimmexc] * (self.params.Ei - self.v[i]) / self.params.CmFS +
                          self.x_nmda(self.v[i]) * self.params.g_nmda * self.GI_nmda[i-self.params.dimmexc] * (self.params.Ee - self.v[i]) / self.params.CmFS +
                          self.params.g_gabaB *
                          self.GI_gabab[i-self.params.dimmexc] *
                          (self.params.EgabaB-self.v[i]) / self.params.CmFS
                          ) * self.stepp * np.heaviside(timet-self.tolsp[i] - self.params.Trefr, 0)

            self.GI_ampa[i-self.params.dimmexc] += self.stepp * \
                (-self.GI_ampa[i-self.params.dimmexc] /
                 self.params.tau_decay_ampa)
            self.GI_nmda[i-self.params.dimmexc] += self.stepp * \
                (-self.GI_nmda[i-self.params.dimmexc] /
                 self.params.tau_decay_nmda)
            self.GI_gabaa[i-self.params.dimmexc] += self.stepp * \
                (-self.GI_gabaa[i-self.params.dimmexc] /
                 self.params.tau_decay_gabaa)

            self.GI_gabab[i-self.params.dimmexc] += self.stepp * (-self.GI_gabab[i-self.params.dimmexc] / self.params.tau_decay_gabaB +
                                                                  self.params.alpha_gabaB * self.QI_gabab[i-self.params.dimmexc])
            self.QI_gabab[i-self.params.dimmexc] += self.stepp * \
                (-self.QI_gabab[i-self.params.dimmexc] /
                 self.params.tau_rise_gabaB)
            '''!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            here the original code breaks as the indices are out of bounds. original code was:
            GI_gabab[i] += stepp * (-GI_gabab[i] / tau_decay_gabaB + alpha_gabaB * QI_gabab[i]) 
            QI_gabab[i] += stepp * (-QI_gabab[i] / tau_rise_gabaB)
            '''

    def update_G_Q(self, timet):
        """ This loop updates conductances of postsynaptic neuron after 'delay' time has passed since a presynaptic neuron has last spiked.
        But maybe the other way aound (pre-post/post-pre)???."""
        for i in range(self.params.dim):
            #  if pre(?)synaptic neuron has spiked and delay has passed
            if (timet - self.spike_flag[i] - self.params.delay) >= 0:
                self.spike_flag[i] = np.inf  #  reset spike flag
                # excitatory neurons as postsynaptic neurons
                for j in range(self.params.self.params.dimmexc):
                    if i < self.params.ne:  #  if presynaptic neuron is excitatory  E -> E
                        self.GE_ampa[j] += self.params.g_ampa * \
                            (self.gij_weights[j][i] *
                             self.gij_connections[j][i])
                        self.GE_nmda[j] += (self.gij_weights[j]
                                            [i] * self.gij_connections[j][i])
                    else:  #  if presynaptic neuron is inhibitory  I -> E
                        self.GE_gabaa[j] += (self.params.g_gabaa *
                                             self.gij_weights[j][i] * self.gij_connections[j][i])
                        self.QE_gabab[j] += self.params.deltaq_gabaB * \
                            self.gij_weights[j][i] * self.gij_connections[j][i]

                # inhibitory neurons as postsynaptic neurons
                for j in range(self.params.diminh):
                    if i < self.params.ne:  #  if presynaptic neuron is excitatory  E -> I
                        self.GI_ampa[j] += self.params.g_ampa * (
                            self.gij_weights[j+self.params.dimmexc][i] * self.gij_connections[j+self.params.dimmexc][i])
                        self.GI_nmda[j] += (self.gij_weights[j+self.params.dimmexc]
                                            [i] * self.gij_connections[j+self.params.dimmexc][i])
                    else:  #  if presynaptic neuron is inhibitory  I -> I
                        self.GI_gabaa[j] += (self.params.g_gabaa * self.gij_weights[j+self.params.dimmexc]
                                             [i] * self.gij_connections[j+self.params.dimmexc][i])
                        self.QI_gabab[j] += self.params.deltaq_gabaB * self.gij_weights[j +
                                                                                        self.params.dimmexc][i] * self.gij_connections[j+self.params.dimmexc][i]

    def STDP(self, timet, dopamine):
        if np.sqrt(dopamine**2) > 5.e-1:
            for i in range(self.params.dimmexc):
                for j in range(self.params.dimmexc):
                    self.eligibility[i][j] = self.eligibility[i][j] * np.exp(
                        (-timet+self.lastupdate[i][j])/self.params.tauc)  # that must be the stdp curve
                    self.lastupdate[i][j] = timet
                    if self.gij_weights[i][j] < self.params.wmax:
                        self.gij_weights[i][j] += 100 * dopamine * \
                            self.eligibility[i][j] * self.stepp
                    if self.gij_weights[i][j] < self.params.wmin:
                        self.gij_weights[i][j] = self.params.wmin

    def on_firing_event(self, timet):
        ''' Save spike info, reset membrane potential, update weights of excitatory (only they are plastic?!)'''
        for i in range(self.params.dim):
            if self.v[i] > self.params.vspike:
                print("spike at time ", timet, " neuron ", i)
                # BOTH excitatory and inhibitory neurons
                self.tolsp[i] = timet  # last spike time for neuron i???
                self.spike_flag[i] = timet  # last spike time for neuron i???
                # ratebinn += 1  # ???
                # reset membrane potential to vres
                self.v[i] = self.params.vres

                # EXCITATORY NEURONS
                if i < self.params.ne:  # if neuron is excitatory
                    rexc += 1  # maybe its the number of spikes in the last sampling period?
                    # ???, bRS is defined as 0 so it resets maybe?
                    self.ww[i] += self.params.bRS

                    # save spike time and neuron index
                    # save spike time and neuron index
                    write_to_file(self.rasterexc, [timet, i])

                    # PLASTICITY
                    for j in range(self.params.dimmexc):  # excitatory neurons
                        if i != j:
                            deltaij = timet - self.tolsp[j]
                            self.eligibility[i][j] = self.eligibility[i][j] * \
                                np.exp(
                                    (-timet+self.lastupdate[i][j])/self.params.tauc)
                            self.lastupdate[i][j] = timet
                            self.eligibility[i][j] += self.params.pp * np.heaviside(
                                self.params.em-self.eligibility[i][j], 0) * np.exp(-deltaij/self.params.taup)
                            self.eligibility[j][i] -= self.params.dd * (
                                self.eligibility[j][i]) * np.exp(-deltaij/self.params.taum)

                # INHIBITORY NEURONS
                else:
                    rinh += 1  # maybe its the number of spikes in the last sampling period?

                    # save spike time and neuron index
                    # save spike time and neuron index
                    write_to_file(self.rasterinh, [timet, i])

    def get_dopamine(self, timet):
        '''
        Computes full dopamine signal for a given time for each rewards. Since rewards are separated in time, 
        we should only consider the closest 1 or 2 (as the response decays really quickly).
        '''
        return (self.params.rewards[0] * doubgauss(timet, self.params.t0+self.params.time_between_rewards*0, self.params.dopamine_T1, self.params.dopamine_T2, self.params.dopamine_amp) +
                self.params.rewards[1] * doubgauss(timet, self.params.t0+self.params.time_between_rewards*1, self.params.dopamine_T1, self.params.dopamine_T2, self.params.dopamine_amp) +
                self.params.rewards[2] * doubgauss(timet, self.params.t0+self.params.time_between_rewards*2, self.params.dopamine_T1, self.params.dopamine_T2, self.params.dopamine_amp) +
                self.params.rewards[3] * doubgauss(timet, self.params.t0+self.params.time_between_rewards*3, self.params.dopamine_T1, self.params.dopamine_T2, self.params.dopamine_amp))

    def time_simulation(self):
        '''
        This function simulates the network for tstop time.
        '''

        # set time variables
        timet = 0  # current time
        told = 0  # last sampling time

        # set progress bar
        pbar = tqdm.tqdm(total=self.tstop, desc='time simulated', unit='s', miniters=.001)

        while timet < self.tstop:
            # 1. Step time
            timet += self.stepp
            pbar.update(self.stepp)  # update progress bar

            # 2. Compute dopamine level at time t
            #dopamine = self.get_dopamine(timet)
            dopamine = 0

            # 3. Sample population activity
            if timet > told + self.sr:  # sr is the sampling rate
                told = timet

                self.sample_population_activity(timet, dopamine)

            # 4. Update biophysical properties of neurons
            self.update_biophysical_properties(timet)

            # 5. Update G and Q, based on E/I type of pre and post synaptic neurons and their connection
            self.update_G_Q(timet)

            # 6. STDP? - update weights based on last update time
            self.STDP(timet, dopamine)

            # 7. Handle firing events
            self.on_firing_event(timet)

        pbar.close()  # close progress bar

        # save weights upon simulation end
        for i in range(self.params.dim):
            for j in range(self.params.dim):
                write_to_file(
                    self.conn_f, [i, j, self.gij_weights[i][j], self.gij_connections[i][j]])

        # close file pointers
        self.rasterexc.close()
        self.rasterinh.close()
        self.timecourse.close()
        self.conn_f.close()

    # Init network parameters

    @staticmethod
    def load_params():
        '''
        This function sets network parameters.
        '''
        params = {
            # Network size parameters (number of neurons)
            'dim': 1000,  # full network size
            'dimmexc': 800,  # excitatory neurons
            'diminh': 200,  # inhibitory neurons
            #'fracin': 0.2,  # fraction of inhibitory neurons
            'ni': int(np.floor(0.2 * 1000)),  # number of inhibitory neurons
            # number of excitatory neurons
            'ne': int(np.floor(1000 * (1 - 0.2))),

            #  Connection parameters
            'pconn': 0.05,
            'muw': 0,
            'sigmaw': 0,
            'wmin': 0.0,  # minimum weight (for the excitatory connections???)
            'wmax': 1.1,  # maximum weight (for the excitatory connections???)

            # Features of external input
            # sequence of rewards: +, +, -, -
            'rewards': np.array([1, 1, -1, -1]),
            'time_between_rewards': 4,  # seconds
            't0': 10,  # time of appearance
            # rise time (seconds)  # this is not used (redefined by hand in the code)
            'dopamine_T1': 0.5,
            # decay time (seconds)  # this is not used (redefined by hand in the code)
            'dopamine_T2': 0.5,
            # amplitude (Hz)  # this is not used (redefined by hand in the code)
            'dopamine_amp': 1,

            # Constant inputs
            # rate of external drive constant in time (Hz) on excitatory neurons
            'inputrate_on_exc0': 1.5,
            'inputrate_on_exc': 1.5,  # why do we need this??
            # rate of external drive constant in time (Hz) on inhibitory neurons
            'inputrate_on_inh': 0,
            'input_extern': 0,  # external input ??
            'input_extern_inh': 0,

            # VARIABLES DEFINITION
            'hui': 0,
            'huj': 0,
            # 'rate': 0,
            # 'ratebinn': 0,
            'randnumb': 0,
            'delay': 0.5 * 1.e-3,
            'kmedio': 0,
            'tauc': 1,  # updates eligibility

            # STDP variables and parameters
            'deltaij': 0,
            'dd': 0.05,
            'pp': 0.1,
            'taup': 10 * 1.e-3,
            'taum': 10 * 1.e-3,
            'em': 10,
            'Ee': 0 * 1.e-3,
            'Ei': -80 * 1.e-3,

            # Neurons parameters
            'glRS': 15. * 1.e-9,
            'CmRS': 200 * 1.e-12,
            'bRS': 0 * 1.e-12,
            'aRS': 0.e-9,
            'tauw': 100 * 1.e-3,
            'deltaRS': 2 * 1.e-3,
            'ElRS': -65 * 1.e-3,

            'ElFS': -65 * 1.e-3,
            'glFS': 15 * 1.e-9,
            'CmFS': 200 * 1.e-12,
            'bFS': 0 * 1.e-12,
            'aFS': 0 * 1.e-9,
            'twFS': 500,
            'deltaFS': 0.5 * 1.e-3,

            # Synaptic dynamics

            # AMPA
            'tau_decay_ampa': 5 * 1.e-3,
            'g_ampa': 1.5 * 1.e-9,

            # GABA_A
            'tau_decay_gabaa': 5 * 1.e-3,
            'g_gabaa': 5 * 1.e-9,

            # NMDA
            'Enmda': 0,
            'tau_decay_nmda': 75 * 1.e-3,
            'g_nmda': 0.012 * 1.e-9,

            # GABAB current
            'EgabaB': -90 * 1.e-3,
            'alpha_gabaB': 1,
            'tau_decay_gabaB': 160 * 1.e-3,
            'tau_rise_gabaB': 90 * 1.e-3,
            'deltaq_gabaB': 0.1,
            'g_gabaB': 1. * 1.e-9,

            'Trefr': 5 * 1.e-3,

            'vres': -65 * 1.e-3,  # reset potential
            'thr': -50 * 1.e-3,
            'vspike': -30 * 1.e-3  # spike threshold
        }

        # convert dictionary to namespace to access parameters as attributes (e.g. params.dim)
        from types import SimpleNamespace

        return SimpleNamespace(**params)

    # Biphys tools
    @staticmethod
    def x_nmda(juy):
        return 1 / (1+1.5 * np.exp(-0.062 * juy)/3.57)

    # Network tools
    @staticmethod
    def init_connections(dim, pconn):
        '''
        Definig random connections with probab 'pconn', stored in matrix 'gij_connections'.

        Connections stores excitatory and inhibitory together.
        '''
        choices = [0, 1]  # 0=not connected, 1=connected
        p_choices = [1-pconn, pconn]  # probability of above values
        gij_connections = np.random.choice(
            choices, size=(dim, dim), p=p_choices)

        return gij_connections

    @staticmethod
    def init_weights(dim, muw, sigmaw):
        '''
        Defining random weights for connections, sampled from a lognormal distribution with mean='muw' and sigma='sigmaw',
        stored in matrix 'gij_weights'.

        Connections stores excitatory and inhibitory together.
        '''

        gij_weights = np.random.lognormal(
            mean=muw, sigma=sigmaw, size=(dim, dim))

        return gij_weights

    @staticmethod
    def limit_excitatory_weights(gij_weights, dimmexc, wmin, wmax):
        '''
        Limit the weights of the excitatory population: wmin < w < wmax.
        '''

        # values outside the interval are clipped to the interval edges.
        gij_weights[:dimmexc, :dimmexc] = np.clip(
            gij_weights[:dimmexc, :dimmexc], wmin, wmax)

        return gij_weights

    # IO tools
    @staticmethod
    def init_file_pointers(savedir):
        """ Init file pointers. """
        import os 

        # init file pointers
        # saves spike times and neuron indices for excitatory neurons
        rasterexc = open(os.path.join(savedir, "rasterplot_EXc_withstimuli_4.csv"), "w")
        # saves spike times and neuron indices for inhibitory neurons
        rasterinh = open(os.path.join(savedir, "rasterplot_Inh_withstimuli_4.csv"), "w")
        # saves timecourse of variables (a lot)
        timecourse = open(os.path.join(savedir, "traces_withstimuli_4.csv"), "w")
        #conn_0 = open("weight_icB.dat","r");
        # saves weights and connections between units (once the simulation's ended)
        conn_f = open(os.path.join(savedir, "weight_fin4.csv"), "w")

        rasterexc.write(",".join(['t', 'unit_id']) + "\n")
        rasterinh.write(",".join(['t', 'unit_id']) + "\n")
        # timecourse.write(f"")  # write header to file: ???
        conn_f.write(",".join(
            ['unit_i', 'unit_j', 'w_ij', 'c_ij']) + "\n")
        
        return rasterexc, rasterinh, timecourse, conn_f
