import math
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, Aer, execute
from qiskit import QuantumRegister, ClassicalRegister
import qiskit.providers.aer as qpa


class QuantumPartonShower_MCM:
    '''
    A class for executing the Quantum Parton Shower (QPS) algorithm, with mid-circuit measurements (MCM).

    Quantum gates controlled on classically processed measurements must be implemented at the hardware level. 
    We can still simulate 'classically-controlled' gates -- but inefficiently -- using the functionality of this class.
    Rather than dynamically selecting which gates on apply via hardware, instead we just apply every possible selection, 
    based on the different measurement possiblities. The h_bool parameter is used throughout the class to implement
    this functionality.

    Note: The second simplification mentioned in the paper -- that we can simply not encode phis and Nones on qubits --
    cannot currently be implemetned in Qiskit. Thus, the particle registers still use 3 qubits.

    Params:
        N  (int): number of steps
        ni (int): number of initial particles
    
    Class attributes:
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
    def __init__(self, N, ni):
        self._N = N
        self._ni = ni
        self._L = int(math.ceil(math.log(N + ni, 2)))

        # Define some register lengths
        self._p_len = 3
        self._h_len = self._L
        self._e_len = 1
        self._w_len = 3

        # Define the registers
        self.pReg, self.hReg, self.w_hReg, self.eReg, self.wReg, self.n_aReg, self.w_aReg= self.allocateQubits()

        self._circuit = QuantumCircuit(self.wReg, self.pReg, self.hReg, self.eReg, self.n_aReg,
                                       self.w_hReg, self.w_aReg)

    def ptype(self, x):
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

    def Delta_bos(self, t, g_a, g_b):
        return math.exp(self.P_bos(t, g_a, g_b))


    def populateParameterLists(self, g_a, g_b, eps):
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
            t_up = eps ** ((i) / self._N)
            t_mid = eps ** ((i + 0.5) / self._N)
            t_low = eps ** ((i + 1) / self._N)
            timeStepList.append(t_mid)
            # Compute values for emission matrices
            Delta_a = self.Delta_f(t_low, g_a) / self.Delta_f(t_up, g_a)
            Delta_b = self.Delta_f(t_low, g_b) / self.Delta_f(t_up, g_b)
            Delta_phi = self.Delta_bos(t_low, g_a, g_b) / self.Delta_bos(t_up, g_a, g_b)
            P_a, P_b, P_phi = self.P_f(t_mid, g_a), self.P_f(t_mid, g_b), self.P_bos(t_mid, g_a, g_b)

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
        w_hReg = QuantumRegister(self._L - 1, 'w_h')
        eReg = QuantumRegister(self._e_len, 'e')
        wReg = QuantumRegister(self._w_len, 'w')
        n_aReg = QuantumRegister(self._L, 'n_a')
        w_aReg = QuantumRegister(self._L - 1, 'w_a')

        return (pReg, hReg, w_hReg, eReg, wReg, n_aReg, w_aReg)


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


    def flavorControl(self, flavor, control, target, ancilla, control_index, target_index, ancilla_index, h_bool=None):
        '''
        Controlled x onto targetQubit if "control" particle is of the correct flavor. 
        
        Params:
            flavor        (str)             which particle type to control on (a, b, or phi)
            control       (QuantumRegister) control particle reg.
            target        (QuantumRegister) 
            ancilla       (QuantumRegister) ancillary reg.
            control_index (int)
            target_index  (int)
            ancilla_index (int)
            h_bool        (List(int))       If none, no .c_if() gates are applied. 
                                            Otherwise, apply apply .c_if() gates for each comp. basis state (integers)
                                            contained in h_bool.
        '''
        if h_bool == None:
            if flavor == "phi":
                self._circuit.x(control[control_index + 1])
                self._circuit.x(control[control_index + 2])
                self._circuit.ccx(control[control_index + 0], control[control_index + 1], ancilla[ancilla_index])
                self._circuit.ccx(control[control_index + 2], ancilla[ancilla_index], target[target_index + 0])
                # undo work
                self._circuit.ccx(control[control_index + 0], control[control_index + 1], ancilla[ancilla_index])
                self._circuit.x(control[control_index + 1])
                self._circuit.x(control[control_index + 2])
            if flavor == "a":
                self._circuit.x(control[control_index + 0])
                self._circuit.ccx(control[control_index + 0], control[control_index + 2], target[target_index + 0])
                # undo work
                self._circuit.x(control[control_index + 0])
            if flavor == "b":
                self._circuit.ccx(control[control_index + 0], control[control_index + 2], target[target_index + 0])
        else:
            for h in h_bool:
                if flavor == "phi":
                    self._circuit.x(control[control_index + 1]).c_if(self.hReg_cl, h)
                    self._circuit.x(control[control_index + 2]).c_if(self.hReg_cl, h)
                    self._circuit.ccx(control[control_index + 0], control[control_index + 1], ancilla[ancilla_index]).c_if(self.hReg_cl, h)
                    self._circuit.ccx(control[control_index + 2], ancilla[ancilla_index], target[target_index + 0]).c_if(self.hReg_cl, h)
                    # undo work
                    self._circuit.ccx(control[control_index + 0], control[control_index + 1], ancilla[ancilla_index]).c_if(self.hReg_cl, h)
                    self._circuit.x(control[control_index + 1]).c_if(self.hReg_cl, h)
                    self._circuit.x(control[control_index + 2]).c_if(self.hReg_cl, h)
                if flavor == "a":
                    self._circuit.x(control[control_index + 0]).c_if(self.hReg_cl, h)
                    self._circuit.ccx(control[control_index + 0], control[control_index + 2], target[target_index + 0]).c_if(self.hReg_cl, h)
                    # undo work
                    self._circuit.x(control[control_index + 0]).c_if(self.hReg_cl, h)
                if flavor == "b":
                    self._circuit.ccx(control[control_index + 0], control[control_index + 2], target[target_index + 0]).c_if(self.hReg_cl, h)


    def incrementer(self, l, b, a, control):
        '''
        Applied a controlled-increment on a given subregister in self.circuit.

        Params:
            l       (int)             length of the counting register
            b       (QuantumRegister) countReg
            a       (QuantumRegister) workReg
            control (Qubit)           control qubit
        '''
        self._circuit.ccx(control, b[0], a[0])
        for j in range(1, l-1):
            self._circuit.ccx(a[j-1], b[j], a[j])

        for j in reversed(range(1, l)):
            self._circuit.cx(a[j-1], b[j])
            if j > 1:
                self._circuit.ccx(a[j-2], b[j-1], a[j-1])
            else:
                self._circuit.ccx(control, b[j-1], a[j-1])

        self._circuit.cx(control, b[0])


    def uCount(self, m, l):
        '''
        Populate the count register (n_aReg) using current particle states.
        Uses wReg[0] as the control and wReg[1] as ancilla qubit for flavorControl and incrementer, respectively.

        Params:
            m (int) iteration/step 
            l (int) current size of n_aReg
        '''
        for k in range(self._ni + m):
            self.flavorControl('a', self.pReg, self.wReg, self.wReg, (k * self._p_len), 0, 1)
            self.incrementer(l, self.n_aReg, self.w_aReg, self.wReg[0])
            self.flavorControl('a', self.pReg, self.wReg, self.wReg, (k * self._p_len), 0, 1)


    def reverse(self, lst):
        ''' Reverse a list (lst) in place. '''
        lst.reverse()
        return lst


    def intToBinary(self, l, number):
        '''
        Converts an integer to a binary list of size l with LSB first and MSB last.
        Each element of the list is an integer (0 or 1), and the list is padded with zeros up to size l.
        
        Params:
            l      (int) size of the binary list
            number (int) to convert
        '''
        numberBinary = [int(x) for x in list('{0:0b}'.format(number))]
        numberBinary = (l - len(numberBinary)) * [0] + numberBinary
        return self.reverse(numberBinary)


    def numberControl(self, l, number, countReg, workReg, h_bool= None):
        '''
        Applies an X to the l-2 (0 indexed) qubit of workReg if countReg encodes the inputted number in binary
        Returns:
            this l-2 qubit, unless l=1, in which case return the only countReg qubit
        DOES NOT CLEAN AFTER ITSELF - USE numberControlT to clean after this operation

        Params:
            l        (int)             size of countReg
            number   (int)             to control on
            countReg (QuantumRegister) reg. on which number is encoded
            workReg  (QuantumRegister) ancillary reg.
            hbool    (List(int))       See flavorControl.
        '''
        if type(number) == int:
            numberBinary = self.intToBinary(l, number)
        else:
            numberBinary = number

        if h_bool == None:
            [self._circuit.x(countReg[i]) for i in range(len(numberBinary)) if numberBinary[i] == 0]
        else:
            for h in h_bool:
                [self._circuit.x(countReg[i]).c_if(self.hReg_cl, h) for i in range(len(numberBinary)) if numberBinary[i] == 0]

        # first level does not use work qubits as control
        if l > 1:
            if h_bool == None: self._circuit.ccx(countReg[0], countReg[1], workReg[0])
            else: 
                for h in h_bool: self._circuit.ccx(countReg[0], countReg[1], workReg[0]).c_if(self.hReg_cl, h)

        # subfunction to recursively handle toffoli gates
        def binaryToffolis(level):
            if h_bool == None: self._circuit.ccx(countReg[level], workReg[level - 2], workReg[level - 1])
            else: 
                for h in h_bool: self._circuit.ccx(countReg[level], workReg[level - 2], workReg[level - 1]).c_if(self.hReg_cl, h)
            if level < l - 1:
                binaryToffolis(level + 1)

        if l > 2:
            binaryToffolis(2)

        # return qubit containing outcome of the operation
        if l == 1:
            return countReg[0]
        else:
            return workReg[l - 2]


    def numberControlT(self, l, number, countReg, workReg, h_bool= None):
        ''' CLEANS AFTER numberControl operation. '''
        if type(number) == int:
            numberBinary = self.intToBinary(l, number)
        else:
            numberBinary = number

        # subfunction to recursively handle toffoli gates
        def binaryToffolisT(level):
            if level < l:
                binaryToffolisT(level + 1)
                if h_bool == None: self._circuit.ccx(countReg[level], workReg[level - 2], workReg[level - 1])
                else: 
                    for h in h_bool: self._circuit.ccx(countReg[level], workReg[level - 2], workReg[level - 1]).c_if(self.hReg_cl, h)

        if l > 2:
            binaryToffolisT(2)

        if l > 1:
            if h_bool == None: self._circuit.ccx(countReg[0], countReg[1], workReg[0])
            else: 
                for h in h_bool: self._circuit.ccx(countReg[0], countReg[1], workReg[0]).c_if(self.hReg_cl, h)

        if h_bool == None:
            [self._circuit.x(countReg[i]) for i in range(len(numberBinary)) if numberBinary[i] == 0]
        else:
            for h in h_bool: [self._circuit.x(countReg[i]).c_if(self.hReg_cl, h) for i in range(len(numberBinary)) if numberBinary[i] == 0]


##############################################################################################################################
# Utilities for classically-controlled gates                                                                                 #
##############################################################################################################################


    @staticmethod
    def history_to_counts(hList, niList):
        '''
        For a given history, encoded as a list hList (output of generate_h), 
        return the total number of particles (n_tot) and the number of bosons (n_phi).

        List index 0 of hList is the first step, and subsequent steps proceed forwards through the list.

        niList is a list of the initial particles, encoding as the following integers:
            5, 7= fermion b
            4, 6= fermion a
            1= boson
            0= no particle
        '''
        fermList= [4, 5, 6, 7]
        ni= len(niList)
        n_tot= ni
        n_phi= 0
        particles= []

        # initialize
        for n in range(ni):
            particles+= [niList[n]]
            if niList[n] == 1:
                n_phi+= 1

        for j in range(len(hList)):
            h= hList[-j-1] # string
            hint= int(h, 2)
            if hint == 0: 
                particles+= [0]
            else:
                if particles[hint-1] in fermList:
                    particles+= [1]
                    n_phi+= 1
                elif particles[hint-1] == 1:
                    particles+= [4]
                    particles[hint-1]= 6
                    n_phi-= 1
                else:
                    print('Warning: emission from None particle. Result: None particle...')
                    particles+= [0]
                n_tot+= 1
        return n_tot, n_phi


    @staticmethod
    def generate_h(N, L, ni, l=[[]]):
        '''
        Generate and return a list of all possible emission histories for an N-step process, starting with ni fermions.

        Params:

            N  (int)  number of steps
            L  (int)  size of counting register
            ni (int)  number of initial particles
            l  (list) of previous histories, empty by default (start at step 0), used recursively

        Returns:
            A list containing all possible emission histories for an N-step shower.
        '''
        if N == 0:
            return [[]]
        nh = len(l[0])
        l_new= []
        for h in l:
            for j in range(nh + ni + 1):
                binj= bin(j)[2:]

                if len(binj) < L:
                    binj= '0'*(L - len(binj)) + binj
                if j < ni + 1: l_new+= [[binj] + h]
                
                # index -1 corresponds to particle ni + 1, index -2 to particle ni + 2
                # For emission on particle j to be possible, particle j can't be zero. And particle j = ni + i
                # corresponds to index -i. ==> index= ni - j

                else:
                    if int(h[ni - j], 2) != 0: l_new+= [[binj] + h]

        if nh >= N - 1:
            return l_new
        else:
            return QuantumPartonShower_MCM.generate_h(N, L, ni, l_new)


    @staticmethod
    def h_map(N, L, niList):
        '''
        Return a map from tuples (n_tot, n_phi) to a list of all possible emission histories (list)
        that could produce that particlar number configuration. This is used to implement .c_if() rotation gates
        by applying a particular rotation (angle), classically controlled on all possible emission histories 
        that induce that angle.

        Note: reverses the order of emissions in the resulting values (lists)

        Note: It's an exponentially long sequence, so this hack does not generalize to arbitrarily large showers.

        Params:
            N      (int)       number of steps
            L      (int)       counting register size
            niList (list(str)) of initial particles
        '''

        histories= QuantumPartonShower_MCM.generate_h(N, L, len(niList))

        counts= {}
        for history in histories:
            c= QuantumPartonShower_MCM.history_to_counts(history, niList)
            if c in counts:
                counts[c]+= [history]
            else:
                counts[c]= [history]

        return counts


##############################################################################################################################
#                                                                                                                            #
##############################################################################################################################


    def uE(self, l, m, Delta_phi, Delta_a, Delta_b, initialParticles):
        '''
        Constructs and applies a quantum circuit that implements uE.
        
        Params:
            l                (int)       length of the counting register
            m                (int)       iteration
            Delta_phi        (float)     Sudakov factor for phis
            Delta_a          (float)     Sudakov factor for a-type fermions
            Delta_b          (float)     Sudakov factor for b-type fermions
            initialParticles (List(str)) used to generate the possible histories.
        '''
        hmap= QuantumPartonShower_MCM.h_map(m, self._L, [int(x, 2) for x in initialParticles])

        for key in hmap:
            n_tot, n_phi= key       

            if m > 0:
                h_bool= [int(''.join(hmap[key][0]), 2)]
                for j in range(1, len(hmap[key])):
                    h_bool+= [int(''.join(hmap[key][j]), 2)]
            else:
                h_bool= None

            for n_a in range(0, n_tot - n_phi + 1):
                n_b= n_tot - n_phi - n_a
                Delta = Delta_phi ** n_phi * Delta_a ** n_a * Delta_b ** n_b
                #print('na= %d, nb= %d, nphi= %d, emit angle: ' %(n_a, n_b, n_phi) + str((2 * math.acos(np.sqrt(Delta)))))
                aControlQub = self.numberControl(l, n_a, self.n_aReg, self.w_aReg, h_bool= h_bool)
                if m > 0: 
                    for h in h_bool: self._circuit.cry((2 * math.acos(np.sqrt(Delta))), aControlQub, self.eReg[0]).c_if(self.hReg_cl, h)
                else: self._circuit.cry((2 * math.acos(np.sqrt(Delta))), aControlQub, self.eReg[0])
                self.numberControlT(l, n_a, self.n_aReg, self.w_aReg, h_bool= h_bool)


    def generateGrayList(self, l, number):
        '''
        Return list of elements in gray code from |0> to |number> where each entry is of type[int, binary list], and
            int:         which bit is the target in the current iteration
            binary list: the state of the rest of the qubits (controls)

        Params:
            l      (int) size of the current count register
            number (int) target for the gray code
        
        Returns:
            the grayList (List(int, List(int)))
        '''
        grayList = [[0, l * [0]]]
        targetBinary = self.intToBinary(l, number)
        for index in range(len(targetBinary)):
            if targetBinary[index] == 1:
                grayList.append([index, (list(grayList[-1][1]))])
                grayList[-1][1][index] = 1
        return grayList[1:]


    def twoLevelControlledRy(self, l, angle, k, externalControl, reg, workReg, h_bool= None):
        '''
        Implements a two level Ry rotation from state |0> to |k>, if externalControl qubit is on.
        For reference: http://www.physics.udel.edu/~msafrono/650/Lecture%206.pdf

        Params:
            l               (int)             rotation register size (# qubits)
            angle           (float)           rotation angle
            k               (int)             integer (comp. basis state) to which this gate rotates
            externalControl (Qubit)           control qubit
            reg             (QuantumRegister) rotation register
            workReg         (QuantumRegister) ancillary
            h_bool          (bool)            See flavorControl.
        '''
        grayList = self.generateGrayList(l, k)
        #print('angle: ' + str(angle))
        if k == 0:
            return
        if l == 1 and k == 1:
            if h_bool == None: self._circuit.cry(angle, externalControl, reg[0])
            else: 
                for h in h_bool: self._circuit.cry(angle, externalControl, reg[0]).c_if(self.hReg_cl, h)
            return
        # swap states according to Gray Code until one step before the end
        for element in grayList:
            targetQub = element[0]
            number = element[1]
            number = number[0:targetQub] + number[targetQub + 1:]
            controlQub = self.numberControl(l - 1, number, reg[0:targetQub] + reg[targetQub + 1:], workReg, h_bool= h_bool)
            if element == grayList[-1]:  # reached end
                if h_bool == None:
                    self._circuit.ccx(controlQub, externalControl, workReg[l - 2])
                    self._circuit.cry(angle, workReg[l - 2], reg[targetQub])
                    self._circuit.ccx(controlQub, externalControl, workReg[l - 2])
                else:
                    for h in h_bool:
                        self._circuit.ccx(controlQub, externalControl, workReg[l - 2]).c_if(self.hReg_cl, h)
                        self._circuit.cry(angle, workReg[l - 2], reg[targetQub]).c_if(self.hReg_cl, h)
                        self._circuit.ccx(controlQub, externalControl, workReg[l - 2]).c_if(self.hReg_cl, h)
            else:  # swap states
                if h_bool == None: self._circuit.cx(controlQub, reg[targetQub])
                else: 
                    for h in h_bool: self._circuit.cx(controlQub, reg[targetQub]).c_if(self.hReg_cl, h)
            self.numberControlT(l - 1, number, reg[0:targetQub] + reg[targetQub + 1:], workReg, h_bool= h_bool)
        # undo
        for element in self.reverse(grayList[:-1]):
            targetQub = element[0]
            number = element[1]
            number = number[0:targetQub] + number[targetQub + 1:]
            controlQub = self.numberControl(l - 1, number, reg[0:targetQub] + reg[targetQub + 1:], workReg, h_bool= h_bool)
            if h_bool == None: self._circuit.cx(controlQub, reg[targetQub])
            else: 
                for h in h_bool: self._circuit.cx(controlQub, reg[targetQub]).c_if(self.hReg_cl, h)
            self.numberControlT(l - 1, number, reg[0:targetQub] + reg[targetQub + 1:], workReg, h_bool= h_bool)
        return


    def U_hAngle(self, flavor, n_phi, n_a, n_b, P_phi, P_a, P_b):
        '''
        Determine angle of rotation used in U_h.

        Params:
            flavor (str)   'phi, 'a', or 'b'
            n_phi  (int)   number of phis
            n_a    (int)   number of a-type fermions
            n_b    (int)   number of b-type fermions
            P_phi  (float) phi splitting probability
            P_a    (float) a-type emission probability
            P_b    (float) b-type emission probability

        Returns:
            rotation angle (float)
        '''
        denominator = n_phi * P_phi + n_a * P_a + n_b * P_b
        if denominator == 0:  # occurs if we are trying the case of no particles remaining (n_a = n_b = n_phi = 0)
            return 0
        flavorStringToP = {'phi': P_phi, 'a': P_a, 'b': P_b}
        emissionAmplitude = np.sqrt(flavorStringToP[flavor] / denominator)
        # correct for arcsin input greater than 1 errors for various input combinations that are irrelevant anyway
        emissionAmplitude = min(1, emissionAmplitude)

        return 2 * np.arcsin(emissionAmplitude)


    def minus1(self, l, countReg, workReg, control):
        '''
        A controlled-decrementor.

        Params:
            l        (int)             size of the countReg
            countReg (QuantumRegister) reg. to decrement
            workReg  (QuantumRegister) ancillary
            control  (Qubit)           
        '''
        [self._circuit.x(qubit) for qubit in countReg]
        self.incrementer(l, countReg, workReg, control)
        [self._circuit.x(qubit) for qubit in countReg]


    def U_h(self, l, m, P_phi, P_a, P_b, initialParticles):
        '''
        Constructs and applies a quantum circuit that implements U_h.
        
        Params:
            l                (int)       size of the counting register
            m                (int)       iteration
            P_phi            (float)     phi splitting probability
            P_a              (float)     a-type emission probability
            P_b              (float)     b-type emission probability
            initialParticles (List(str))
        '''
        for k in reversed(range(self._ni + m)):
            if k >= self._ni:
                hmap= QuantumPartonShower_MCM.h_map(k - self._ni + 1, self._L, [int(x, 2) for x in initialParticles])
            else:
                hmap= QuantumPartonShower_MCM.h_map(0, self._L, [int(x, 2) for x in initialParticles[:k+1]])

            for key in hmap:
                n_tot, n_phi= key
                if m == 0 or k < self._ni:
                    h_bool= None
                else:
                    h_bool= [int(''.join(hmap[key][0]), 2)]
                    for j in range(1, len(hmap[key])):
                        h_bool+= [int(''.join(hmap[key][j]), 2)]

                for n_a in range(0, n_tot - n_phi + 1):
                    #print('\n')
                    n_b= n_tot - n_phi - n_a
                    aControl = self.numberControl(l, n_a, self.n_aReg, self.w_aReg, h_bool= h_bool)
                    # controlled R-y from |0> to |k> on all qubits with all possible angles depending on n_phi, n_a, n_b, and flavor
                    for flavor in ['phi', 'a', 'b']:
                        angle = self.U_hAngle(flavor, n_phi, n_a, n_b, P_phi, P_a, P_b)
                        #print('na= %d, nb= %d, nphi= %d, flavor: ' %(n_a, n_b, n_phi) + flavor + ', angle= ' + str(angle))
                        self.flavorControl(flavor, self.pReg, self.wReg, self.wReg, (k * self._p_len), 0, 2, h_bool= h_bool)  # wReg[2] is work qubit but is reset to 0
                        if m == 0 or k < self._ni:
                            self._circuit.ccx(aControl, self.wReg[0], self.wReg[1])
                            self._circuit.ccx(self.eReg[0], self.wReg[1], self.wReg[2])
                        else:
                            for h in h_bool:
                                self._circuit.ccx(aControl, self.wReg[0], self.wReg[1]).c_if(self.hReg_cl, h)
                                self._circuit.ccx(self.eReg[0], self.wReg[1], self.wReg[2]).c_if(self.hReg_cl, h)
                        self.twoLevelControlledRy(l, angle, k + 1, self.wReg[2], self.hReg, self.w_hReg, h_bool= h_bool)

                        if m == 0 or k < self._ni:
                            self._circuit.ccx(self.eReg[0], self.wReg[1], self.wReg[2])  # next steps undo work qubits
                            self._circuit.ccx(aControl, self.wReg[0], self.wReg[1])
                        else:
                            for h in h_bool:
                                self._circuit.ccx(self.eReg[0], self.wReg[1], self.wReg[2]).c_if(self.hReg_cl, h)  # next steps undo work qubits
                                self._circuit.ccx(aControl, self.wReg[0], self.wReg[1]).c_if(self.hReg_cl, h)

                        self.flavorControl(flavor, self.pReg, self.wReg, self.wReg, (k * self._p_len), 0, 2, h_bool= h_bool)  # wReg[2] is work qubit but is reset to 0
                    self.numberControlT(l, n_a, self.n_aReg, self.w_aReg, h_bool= h_bool)

            # subtract from the counts register depending on which flavor particle emitted
            self.flavorControl('a', self.pReg, self.wReg, self.wReg, (k * self._p_len), 0, 1)  # wReg[2] is work qubit but is reset to 0
            self.minus1(l, self.n_aReg, self.w_aReg, self.wReg[0])
            self.flavorControl('a', self.pReg, self.wReg, self.wReg, (k * self._p_len), 0, 1)  # wReg[2] is work qubit but is reset to 0

        # apply x on eReg if hReg[m] = 0, apply another x so we essentially control on not 0 instead of 0
        isZeroControl = self.numberControl(l, 0, self.hReg, self.w_hReg)
        self._circuit.cx(isZeroControl, self.eReg[0])
        self._circuit.x(self.eReg[0])
        self.numberControlT(l, 0, self.hReg, self.w_hReg)


    def U_p(self, l, m, g_a, g_b):
        '''
        Constructs and applies a quantum circuit that implements U_p.

        Params:
            l   (int)   size of the counting & history registers
            m   (int)   iteration
            g_a (float) a-type coupling
            g_b (float) b-type coupling
        '''
        for k in range(0, self._ni + m):
            pk0= k * self._p_len # particle k first (zero) index
            pNew= (self._ni + m) * self._p_len # new/current particle first(zero) index

            # Have to get all histories where the current emission is from particle k.
            gen_h= QuantumPartonShower_MCM.generate_h(m+1, self._L, self._ni) # current particle to update is actually step m+1
            gen_hk= []
            for h in gen_h:
                if int(h[0], 2) == k + 1: # k + 1 is the correct particle
                    gen_hk+= [h]

            for hList in gen_hk:
                h= int(''.join(hList), 2)
                #print(h)
                self._circuit.cx(self.pReg[pk0 + 2], self.pReg[pNew + 0]).c_if(self.hReg_cl, h)

                # second gate in paper (undoes work register immediately)
                self._circuit.x(self.pReg[pk0 + 1]).c_if(self.hReg_cl, h)
                self._circuit.x(self.pReg[pk0 + 2]).c_if(self.hReg_cl, h)
                self._circuit.cx(self.pReg[pk0 + 2], self.wReg[0]).c_if(self.hReg_cl, h)
                self._circuit.ccx(self.wReg[0], self.pReg[pk0 + 1], self.wReg[1])
                self._circuit.ccx(self.wReg[1], self.pReg[pk0 + 0], self.pReg[pNew + 2])
                self._circuit.ccx(self.wReg[0], self.pReg[pk0 + 1], self.wReg[1])
                self._circuit.cx(self.pReg[pk0 + 2], self.wReg[0]).c_if(self.hReg_cl, h)
                self._circuit.x(self.pReg[pk0 + 1]).c_if(self.hReg_cl, h)
                self._circuit.x(self.pReg[pk0 + 2]).c_if(self.hReg_cl, h)

                # third gate in paper
                self._circuit.cx(self.pReg[pNew + 2], self.pReg[pk0 + 2]).c_if(self.hReg_cl, h)

                # fourth and fifth gate in paper (then undoes work register)
                self._circuit.ch(self.pReg[pNew + 2], self.pReg[pNew + 1]).c_if(self.hReg_cl, h)
                angle = (2 * np.arccos(g_a / np.sqrt(g_a ** 2 + g_b ** 2)))
                self._circuit.cry(angle, self.pReg[pNew + 2], self.pReg[pNew + 0]).c_if(self.hReg_cl, h)

                # sixth and seventh gate in paper (then undoes work register)
                self._circuit.x(self.pReg[pNew + 0]).c_if(self.hReg_cl, h)
                self._circuit.x(self.pReg[pNew + 1]).c_if(self.hReg_cl, h)
                self._circuit.cx(self.pReg[pNew + 1], self.wReg[0]).c_if(self.hReg_cl, h)
                self._circuit.ccx(self.wReg[0], self.pReg[pNew + 2], self.pReg[pk0 + 1])
                self._circuit.cx(self.pReg[pNew + 1], self.wReg[0]).c_if(self.hReg_cl, h)
                self._circuit.cx(self.pReg[pNew + 0], self.wReg[0]).c_if(self.hReg_cl, h)
                self._circuit.ccx(self.wReg[0], self.pReg[pNew + 2], self.pReg[pk0 + 0])
                self._circuit.cx(self.pReg[pNew + 0], self.wReg[0]).c_if(self.hReg_cl, h)
                self._circuit.x(self.pReg[pNew + 0]).c_if(self.hReg_cl, h)
                self._circuit.x(self.pReg[pNew + 1]).c_if(self.hReg_cl, h)


    def createCircuit(self, eps, g_1, g_2, g_12, initialParticles, verbose=False):
        '''
        This is the main function to create a quantum circuit that implements QPS. 
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
        # calculate constants
        gp = math.sqrt(abs((g_1 - g_2) ** 2 + 4 * g_12 ** 2))
        if g_1 > g_2:
            gp = -gp
        g_a, g_b = (g_1 + g_2 - gp) / 2, (g_1 + g_2 + gp) / 2
        u = math.sqrt(abs((gp + g_1 - g_2) / (2 * gp)))

        # evaluate P(Theta) and Delta(Theta) at every time step
        self.populateParameterLists(g_a, g_b, eps)

        qubits = {'pReg': self.pReg, 'hReg': self.hReg, 'w_hReg': self.w_hReg, 'eReg': self.eReg, 'wReg': self.wReg,
                  'n_aReg': self.n_aReg, 'w_aReg': self.w_aReg}

        self.initializeParticles(initialParticles)

        (self.wReg_cl, self.pReg_cl, self.hReg_cl, self.eReg_cl, self.n_aReg_cl, self.w_hReg_cl, self.w_aReg_cl) = self.allocateClbits()
        
        self.add_Clbits()

        # begin stepping through subcircuits
        for m in range(self._N):
            if verbose:
                print('\n\nm= %d\n\n' %(m))
            l = int(math.floor(math.log(m + self._ni, 2)) + 1)

            # R^(m) - rotate every particle p_k from 1,2 to a,b basis (step 1)
            index = 0
            while index < self.pReg.size:
                self._circuit.cry((2 * math.asin(-u)), self.pReg[index + 2], self.pReg[index + 0])
                index += self._p_len

            # populate count register (step 2)
            if verbose:
                print('Apply uCount()...')
            self.uCount(m, l)

            # assess if emmision occured (step 3)
            if verbose:
                print('Apply uE()...')
            self.uE(l, m, self.Delta_phiList[m], self.Delta_aList[m], self.Delta_bList[m], initialParticles)

            if verbose:
                print('Apply U_h()...')
            self.U_h(l, m, self.P_phiList[m], self.P_aList[m], self.P_bList[m], initialParticles)

            if verbose:
                print('Measure and reset |h>...')
            self._circuit.measure(self.hReg, self.hReg_cl[m*self._L : (m+1)*self._L])
            self._circuit.reset(self.hReg)

            if verbose:
                print('Apply U_p()...')
            # update particle based on which particle split/emmitted (step 5)
            self.U_p(l, m, g_a, g_b)

            # R^-(m) rotate every particle p_k from a,b to 1,2 basis (step 6)
            index2 = 0
            while index2 < self.pReg.size:
                self._circuit.cry((2 * math.asin(u)), self.pReg[index2 + 2], self.pReg[index2 + 0])
                index2 += self._p_len

        if verbose:
            print('generated circuit on', self._circuit.num_qubits, 'qubits')
        
        return self._circuit, qubits


    def allocateClbits(self):
        ''' Generate and return all classical registers. '''
        nbits_h = self._N * self._L

        wReg_cl = ClassicalRegister(3, 'w_cl')
        pReg_cl = []
        for j in range(self._N + self._ni):
            pReg_cl.append(ClassicalRegister(3, 'p%d_cl' %(j)))
        hReg_cl = ClassicalRegister(nbits_h, 'h_cl')
        eReg_cl = ClassicalRegister(self._e_len, 'e_cl')
        n_aReg_cl = ClassicalRegister(self._L, 'na_cl')

        w_hReg_cl = ClassicalRegister(self._L - 1, 'wh_cl')
        w_aReg_cl = ClassicalRegister(self._L - 1, 'wa_cl')

        return (wReg_cl, pReg_cl, hReg_cl, eReg_cl, n_aReg_cl, w_hReg_cl, w_aReg_cl)


    def add_Clbits(self):
        ''' Add all classical registers to self._circuit. '''
        self._circuit.add_register(self.wReg_cl)
        for j in range(self._N + self._ni):
            self._circuit.add_register(self.pReg_cl[j])
        self._circuit.add_register(self.hReg_cl)
        self._circuit.add_register(self.eReg_cl)
        self._circuit.add_register(self.n_aReg_cl)
        self._circuit.add_register(self.w_hReg_cl)
        self._circuit.add_register(self.w_aReg_cl)


    def measure_Clbits(self):
        ''' Measure all qubits other than the history register (already measured mid-circuit). '''
        self._circuit.measure(self.wReg, self.wReg_cl)
        for j in range(self._N + self._ni):
            self._circuit.measure(self.pReg[3*j : 3*(j+1)], self.pReg_cl[j])

        self._circuit.measure(self.eReg, self.eReg_cl)
        self._circuit.measure(self.n_aReg, self.n_aReg_cl)
        self._circuit.measure(self.w_hReg, self.w_hReg_cl)
        self._circuit.measure(self.w_aReg, self.w_aReg_cl)


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
            simulator = Aer.get_backend('qasm_simulator')
            self.measure_Clbits()
            job = execute(self._circuit, simulator, shots=shots)
            result = job.result()
            counts = result.get_counts(self._circuit)
            return counts

        elif type == 'mps':
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
            print("Invalid simulator. For the first paramter, choose 'qasm', 'mps', or 'statevector'.")