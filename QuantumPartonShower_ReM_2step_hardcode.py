import math
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, Aer, execute
from qiskit import QuantumRegister, ClassicalRegister
import qiskit.providers.aer as qpa


class QuantumPartonShower:
    '''
    A class for executing a 2-step Quantum Parton Shower (QPS) algorithm, with mid-circuit measurements.

    The main quantum circuit is constructed by hand-optimizing the general circuit constructed as
    in the paper. This results in a siginificant discount, compared to quoted gate counts. In fact,
    the circuit constructed here uses:
        12 qubits
        18 CNOTs
        50 single-qubit gates (1 H, 17 X, 32 generic)
        3 qubit resets
        14 qubit measurements
        
        Gates: OrderedDict([('x', 20), ('cu3', 8), ('measure', 5), ('reset', 3), ('cry', 3), ('cx', 2), ('ry', 1), ('h', 1), ('u3', 1)])
    
    Params:
        g_1  (float) type-1 fermion coupling
        g_2  (float) type-2 fermion coupling
        g_12 (float) mixed fermion coupling
        eps  (float) scale parameter

    Class attributes:
        g_1, g_2, g_12, eps

        N        (int)             number of simulation steps
        ni       (int)             number of initial particles
        L        (int)             ceil(log2(N+ni))
        p_len    (int)             number of qubits per particle
        h_len    (int)             number of qubits for each history step
        e_len    (int)             number of qubits in the emission register
        w_len    (int)             number of qubits in the main ancillary register

        pReg     (QuantumRegister) particle register
        hReg     (QuantumRegister) history register
        w_hReg   (QuantumRegister) ancillary reg. for Uh
        eReg     (QuantumRegister) emission qubit
        wReg     (QuantumRegister) main ancillary reg.
        n_aReg   (QuantumRegister) number register
        w_aReg   (QuantumRegister) ancillary reg. for Ucount
        circuit  (QuantumCircuit)  that implements QPS
    '''
    def __init__(self, g_1, g_2, g_12, eps):
        self._N = 2
        self._ni = 1
        self.g_1= g_1
        self.g_2= g_2
        self.g_12= g_12
        self.eps= eps

        # Derived params
        self._L = int(math.ceil(math.log(self._N + self._ni, 2)))

        gp = math.sqrt(abs((g_1 - g_2) ** 2 + 4 * g_12 ** 2))
        if g_1 > g_2:
            gp = -gp
        self.g_a= (g_1 + g_2 - gp) / 2
        self.g_b= (g_1 + g_2 + gp) / 2
        self.u = math.sqrt(abs((gp + g_1 - g_2) / (2 * gp)))
        
        # Define these variables for indexing - to convert from cirq's grid qubits (see explaination in notebook)
        self._p_len = 3
        self._h_len = self._L
        self._e_len = 1
        self._w_len = 3

        #defining the registers
        self.pReg, self.hReg, self.eReg= self.allocateQubits()

        self._circuit = QuantumCircuit(self.pReg, self.hReg, self.eReg)

    def __str__(self):
        tot_qubits= 3*(self._N + self._ni) + self._L + 1
        return "N= %d \n ni= %d \n L= %d \n Total qubits: %d" %(self._N, self._ni, self._L, tot_qubits)

    @staticmethod
    def ptype(x):
        ''' Parses particle type, from binary string to descriptive string. '''
        if x=='000':
            return '0'
        if x=='001':
            return 'phi'
        if x=='100':
            return 'f1'
        if x=='101':
            return 'f2'
        if x=='110':
            return 'af1'
        if x=='111':
            return 'af2'
        else:
            return 'NAN'

    def P_f(self, t, g):
        alpha = g ** 2 * self.Phat_f(t) / (4 * math.pi)
        return alpha

    def Phat_f(self, t):
        return math.log(t)

    def Phat_bos(self, t):
        return math.log(t)

    def Delta_f(self, t, g):
        return math.exp(self.P_f(t, g))

    def P_bos(self, t, g_a, g_b):
        alpha = g_a ** 2 *self.Phat_bos(t) / (4 * math.pi) + g_b ** 2 * self.Phat_bos(t) / (4 * math.pi)
        return alpha

    def P_bos_g(self, t, g):
        return g ** 2 *self.Phat_bos(t) / (4 * math.pi)

    def Delta_bos(self, t, g_a, g_b):
        return math.exp(self.P_bos(t, g_a, g_b))


    def populateParameterLists(self):
        '''
        Populates the 6 parameter lists -- 3 splitting functions, 3 Sudakov factors -- 
        with correct values for each time step theta.

        Params:
            g_a           (float)       a-coupling
            g_b           (float)       b-coupling
            eps           (float)       discretization parameter
            
        Sets the following attributes:
            timeStepList  (List(float)) scale / opening angle array
            P_aList       (List(float)) Splitting functions for a-type fermions
            P_bList       (List(float)) SPlitting functions for b-type fermions
            P_phiList     (List(float)) Splitting functions for phis
            Delta_aList   (List(float)) Sudakov factors for a-type fermions
            Delta_bList   (List(float)) Sudakov factors for b-type fermions
            Delta_phiList (List(float)) Sudakov factors for phis
        '''
        timeStepList, P_aList, P_bList, P_phiList, Delta_aList, Delta_bList, Delta_phiList= [], [], [], [], [], [], []
        for i in range(self._N):
            # Compute time steps
            t_up = self.eps ** ((i) / self._N)
            t_mid = self.eps ** ((i + 0.5) / self._N)
            t_low = self.eps ** ((i + 1) / self._N)
            timeStepList.append(t_mid)
            # Compute values for emission matrices
            Delta_a = self.Delta_f(t_low, self.g_a) / self.Delta_f(t_up, self.g_a)
            Delta_b = self.Delta_f(t_low, self.g_b) / self.Delta_f(t_up, self.g_b)
            Delta_phi = self.Delta_bos(t_low, self.g_a, self.g_b) / self.Delta_bos(t_up, self.g_a, self.g_b)
            P_a, P_b, P_phi = self.P_f(t_mid, self.g_a), self.P_f(t_mid, self.g_b), self.P_bos(t_mid, self.g_a, self.g_b)

            # Add them to the list
            P_aList.append(P_a)
            P_bList.append(P_b)
            P_phiList.append(P_phi)
            Delta_aList.append(Delta_a)
            Delta_bList.append(Delta_b)
            Delta_phiList.append(Delta_phi)

        self.timeStepList= timeStepList
        self.P_aList= P_aList
        self.P_bList= P_bList
        self.P_phiList= P_phiList
        self.Delta_aList= Delta_aList
        self.Delta_bList= Delta_bList
        self.Delta_phiList= Delta_phiList


    def allocateQubits(self):
        ''' Create and return all quantum registers needed. '''
        nqubits_p = self._p_len * (self._N + self._ni)

        pReg = QuantumRegister(nqubits_p, 'p')
        hReg = QuantumRegister(self._L, 'h')
        eReg = QuantumRegister(1, 'e')

        return (pReg, hReg, eReg)


    def initializeParticles(self, initialParticles):
        ''' 
        Apply appropriate X gates to ensure that the p register contains all of the initial particles.
            The p registers contains particles in the form of a string '[MSB, middle bit, LSB]'.

        Params:
            initialParticles (List(str)) of the initial particles, where each particle is a binary string (see ptype)    
        '''
        for currentParticleIndex in range(len(initialParticles)):
            for particleBit in range(3):
                pBit= 2 - particleBit
                if int(initialParticles[currentParticleIndex][particleBit]) == 1:
                    self._circuit.x(self.pReg[currentParticleIndex * self._p_len + pBit])


    def createCircuit(self, initialParticles, verbose=False):
        '''
        This is the main function to create a quantum circuit that implements a 2-step QPS with mid-circuit measurements. 
        The circuit is constructed in place (self._circuit), and also returned.

        Params:
            eps              (float)     scale parameter
            g_1              (float)     type-1 fermion coupling
            g_2              (float)     type-2 fermion coupling
            g_12             (float)     mixed fermion coupling
            initialParticles (List(str)) list of initial particles, each represented by its binary string: 'MSB, middle bit, LSB'
            verbose          (bool)      whether to print out useful info while constructing the circuit

        Returns:
            a tuple: QPS circuit (QuantumCircuit), qubit dict (dict)
        '''
        # evaluate P(Theta) and Delta(Theta) at every time step
        self.populateParameterLists()

        if verbose:
            print('Classical parameters: \n')
            print('g_a= %.4f, g_b= %.4f, u= %.4f' %(self.g_a, self.g_b, self.u))
            print('Delta_aList: ' + str(Delta_aList))
            print('Delta_bList: ' + str(Delta_bList))
            print('Delta_phiList: ' + str(Delta_phiList))
            print('P_aList: ' + str(P_aList))
            print('P_bList: ' + str(P_bList))
            print('P_phiList: ' + str(P_phiList))
            print('timeStepList: ' + str(timeStepList))
            print('\n')

        qubits = {'pReg': self.pReg, 'hReg': self.hReg, 'eReg': self.eReg}

        self.initializeParticles(initialParticles)

        (self.pReg_cl, self.hReg_cl) = self.allocateClbits()

        self.add_Clbits()

        #########################################################################################################
        # Step 1                                                                                                #
        #########################################################################################################
        print('Applying step 1.')

        # R^(m) - rotate every particle p_k from 1,2 to a,b basis (step 1)
        self._circuit.ry((2 * math.asin(-self.u)), self.pReg[0])

        # assess if emmision occured (step 3)
        if verbose:
            print('\n Apply uE()...')
            print('\t DeltaA: ' + str(Delta_aList[0]))
            print('\t DeltaB: ' + str(Delta_bList[0]))
        self._circuit.x(self.pReg[0])
        self._circuit.cu3(2*np.arccos(math.sqrt(self.Delta_aList[0])), 0, 0, self.pReg[0], self.eReg[0]) #a emission rotation
        self._circuit.x(self.pReg[0])
        self._circuit.cu3(2*np.arccos(math.sqrt(self.Delta_bList[0])), 0, 0, self.pReg[0] , self.eReg[0]) #b emission rotation

        if verbose:
            print('Measure and Reset |e>...')
        self._circuit.measure(self.eReg, self.hReg_cl[0][-1])
        self._circuit.reset(self.eReg)
        
        if verbose:
            print('Apply U_h()...')
        #########################################################################################################
        #self._circuit.u3(2*np.arccos(0), 0, 0, self.hReg[0]).c_if(self.hReg_cl[0], 2**self._L)
        self._circuit.x(self.hReg[0]).c_if(self.hReg_cl[0], 2**self._L)
        #########################################################################################################

        if verbose:
            print('Measure and reset |h>...')
        # NOTE: ONLY NEED TO MEASURE AND RESET hReg[0]
        #self._circuit.measure(self.hReg, self.hReg_cl[0][:self._L])
        #self._circuit.reset(self.hReg)
        self._circuit.measure(self.hReg[0], self.hReg_cl[0][0])
        self._circuit.reset(self.hReg[0])

        if verbose:
            print('Apply U_p()...') # update particle based on which particle split/emmitted (step 5)
        self._circuit.x(self.pReg[3]).c_if(self.hReg_cl[0], 5)

        
        #########################################################################################################
        # Step 2                                                                                                #
        #########################################################################################################
        if True:
            print('Applying step 2.')

            if verbose:
                print('\n Apply uE()...')
                print('\t DeltaAphi: ' + str(self.Delta_phiList[1] * self.Delta_aList[1]))
                print('\t DeltaBphi: ' + str(self.Delta_phiList[1] * self.Delta_bList[1]))
                print('\t DeltaA: ' + str(self.Delta_aList[1]))
                print('\t DeltaB: ' + str(self.Delta_bList[1]))
            self._circuit.x(self.pReg[0])
            self._circuit.cu3(2*np.arccos(math.sqrt(self.Delta_phiList[1] * self.Delta_aList[1])), 0, 0, self.pReg[0], self.eReg[0]).c_if(self.hReg_cl[0], 5)
            self._circuit.x(self.pReg[0])
            self._circuit.cu3(2*np.arccos(math.sqrt(self.Delta_phiList[1] * self.Delta_bList[1])), 0, 0, self.pReg[0], self.eReg[0]).c_if(self.hReg_cl[0], 5)

            if verbose:
                print('a-phi emit angle: ' + str(2*np.arccos(math.sqrt(self.Delta_phiList[1] * self.Delta_aList[1]))))
                print('b-phi emit angle: ' + str(2*np.arccos(math.sqrt(self.Delta_phiList[1] * self.Delta_bList[1]))))
                print('a emit angle: ' + str(2*np.arccos(math.sqrt(self.Delta_aList[1]))))
                print('b emit angle: ' + str(2*np.arccos(math.sqrt(self.Delta_bList[1]))))
            self._circuit.x(self.pReg[0])
            self._circuit.cu3(2*np.arccos(math.sqrt(self.Delta_aList[1])), 0, 0, self.pReg[0], self.eReg[0]).c_if(self.hReg_cl[0], 0)
            self._circuit.x(self.pReg[0])
            self._circuit.cu3(2*np.arccos(math.sqrt(self.Delta_bList[1])), 0, 0, self.pReg[0], self.eReg[0]).c_if(self.hReg_cl[0], 0)

            if verbose:
                print('Measure and reset |e>...')
            self._circuit.measure(self.eReg, self.hReg_cl[0][-1])

            if verbose:
                print('Apply U_h()...')
            # Over p0
            t_mid= self.timeStepList[1]
            entry_h_a = 0
            entry_h_aphi = math.sqrt(1-(self.P_f(t_mid, self.g_a)/(self.P_f(t_mid, self.g_a) + self.P_bos(t_mid, self.g_a, self.g_b)))) #off diagonals in A23
            entry_h_b = 0
            entry_h_bphi = math.sqrt(1-(self.P_f(t_mid, self.g_b)/(self.P_f(t_mid, self.g_b) + self.P_bos(t_mid, self.g_a, self.g_b))))

            self._circuit.x(self.pReg[0])
            #########################################################################################################
            #v1:self._circuit.cu3(2*np.arccos(entry_h_a), 0, 0, self.pReg[0], self.hReg[0]).c_if(self.hReg_cl[0], 4)
            #v2: self._circuit.cx(self.pReg[0], self.hReg[0]).c_if(self.hReg_cl[0], 4)
            self._circuit.x(self.hReg[0]).c_if(self.hReg_cl[0], 4)
            #########################################################################################################
            
            self._circuit.cu3(2*np.arccos(entry_h_aphi), 0, 0, self.pReg[0], self.hReg[0]).c_if(self.hReg_cl[0], 5)
            self._circuit.x(self.pReg[0])
            
            #########################################################################################################
            #v1: self._circuit.cu3(2*np.arccos(entry_h_b), 0, 0, self.pReg[0], self.hReg[0]).c_if(self.hReg_cl[0], 4)
            #v2: self._circuit.cx(self.pReg[0], self.hReg[0]).c_if(self.hReg_cl[0], 4)
            # current: combined with the cx a few lines up
            #########################################################################################################
            
            self._circuit.cu3(2*np.arccos(entry_h_bphi), 0, 0, self.pReg[0], self.hReg[0]).c_if(self.hReg_cl[0], 5)

            self._circuit.measure(self.hReg[0], self.hReg_cl[1][0])
            self._circuit.measure(self.eReg, self.hReg_cl[1][-1])
            self._circuit.reset(self.eReg)

            # Now over p1
            entry_h_phi = 0

            #########################################################################################################
            #v1: self._circuit.cu3(2*np.arccos(entry_h_phi), 0, 0, self.pReg[3], self.hReg[1]).c_if(self.hReg_cl[1], 4)
            #v2: self._circuit.cx(self.pReg[3], self.hReg[1]).c_if(self.hReg_cl[1], 4)
            self._circuit.x(self.hReg[1]).c_if(self.hReg_cl[1], 4)
            #########################################################################################################
            
            # NOTE: WE DON'T NEED TO MEASURE hReg[1] TO GET WHAT WE WANT. IN FACT WE REALLY ONLY NEED 2 
            #       CLASSICAL BITS TO CONDITION ON
            #self._circuit.measure(self.hReg[1], self.hReg_cl[1][1])

            if verbose:
                print('Apply U_p()...')
            self._circuit.x(self.pReg[6]).c_if(self.hReg_cl[1], 5)
            self._circuit.x(self.pReg[8]).c_if(self.hReg_cl[1], 4)
            self._circuit.x(self.pReg[5]).c_if(self.hReg_cl[1], 4)

            self._circuit.h(self.pReg[7]).c_if(self.hReg_cl[1], 4)
            entry_r = self.g_a / (math.sqrt(self.g_a*self.g_a + self.g_b*self.g_b))

            self._circuit.u3(2*np.arccos(entry_r), 0, 0, self.pReg[6]).c_if(self.hReg_cl[1], 4)
            self._circuit.x(self.pReg[7])
            self._circuit.cx(self.pReg[7], self.pReg[4]).c_if(self.hReg_cl[1], 4)
            self._circuit.x(self.pReg[7])
            self._circuit.x(self.pReg[6])
            self._circuit.cx(self.pReg[6], self.pReg[3]).c_if(self.hReg_cl[1], 4)
            self._circuit.x(self.pReg[6])

        # R^-(m) rotate every particle p_k from a,b to 1,2 basis (step 6)
        index2 = 0
        while index2 < self.pReg.size:
            # circuit.append(ry(2*math.asin(u)).controlled().on(p_k[2], p_k[0]))
            self._circuit.cry((2 * math.asin(self.u)), self.pReg[index2 + 2], self.pReg[index2 + 0])
            index2 += self._p_len

        print('Done.\n')
        return self._circuit, qubits


    def allocateClbits(self):
        '''
        Generate and return all classical registers.

        Note: We have to measure |h> and |e> on the same register to have to proper classical controls.
        '''
        nbits_h = self._N * self._L

        pReg_cl = []
        for j in range(self._N + self._ni):
            pReg_cl.append(ClassicalRegister(3, 'p%d_cl' %(j)))

        # We extend each hReg_cl[i] by one qubit. This extra qubit is a place to measure |e>, if needed there.
        hReg_cl = []
        for j in range(self._N):
            hReg_cl.append(ClassicalRegister(self._L + 1, 'h%d_cl' %(j)))

        return (pReg_cl, hReg_cl)


    def add_Clbits(self):
        ''' Add all classical registers to self._circuit. '''
        for j in range(self._N + self._ni):
            self._circuit.add_register(self.pReg_cl[j])
        for j in range(self._N):
            self._circuit.add_register(self.hReg_cl[j])


    def measure_Clbits(self):
        ''' Measure all qubits other than the history register (already measured mid-circuit). '''
        for j in range(self._N + self._ni):
            self._circuit.measure(self.pReg[3*j : 3*(j+1)], self.pReg_cl[j])


    def simulate(self, type, shots=None, position=False):
        '''
        Simulate self.circuit using the specified simulator.

        Params:
            type     (str)  Either the qasm simulator or the statevector simulator
            shots    (int)  If using the qasm simulator the number of shots needs to be specified
            position (bool) the statevector is very long, so if position=True the function will print the value and
                            position of tbe non-zero elements
        Returns: 
            execute().result().get_counts() for qasm or mps simulation, otherwise
            execute().result().get_statevector() for statevector simulation
        '''
        if type == 'qasm':
            #simulator = Aer.get_backend('qasm_simulator')
            #simulator = Aer.get_backend('aer_simulator_matrix_product_state')
            simulator = qpa.QasmSimulator(method= 'matrix_product_state')

            self.measure_Clbits()
            job = execute(self._circuit, simulator, shots=shots)
            result = job.result()
            counts = result.get_counts(self._circuit)
            return counts

        elif type == 'statevector':
            simulator = Aer.get_backend('statevector_simulator')
            result = execute(self._circuit, simulator).result()
            statevector = result.get_statevector(self._circuit)
            if position:
                [print("position of non zero element: ", list(statevector).index(i), "\nvalue: ",
                       i, "\nabsolute value: ", abs(i)) for i in statevector if abs(i) > 10 ** (-5)]
            return statevector
        else:
            print("choose 'qasm' or 'statevector'")