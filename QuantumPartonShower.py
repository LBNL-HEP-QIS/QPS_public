import math
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, Aer, execute
from qiskit import QuantumRegister, ClassicalRegister
import qiskit.providers.aer as qpa


class QuantumPartonShower:
    '''
    A class for executing the Quantum Parton Shower (QPS) algorithm.

    Params:
        N    (int)   number of steps
        ni   (int)   number of initial particles
        g_1  (float) type-1 fermion coupling
        g_2  (float) type-2 fermion coupling
        g_12 (float) mixed fermion coupling
        eps  (float) scale parameter
    
    Class attributes:
        N, ni, g_1, g_2, g_12, eps

        L        (int)   ceil(log2(N+ni))
        g_a      (float) a-type fermion coupling (eq. 6)
        g_b      (float) b-type fermion coupling (eq. 6)
        u        (float) rotation angle for diagonalization (eq. 7)
        p_len    (int)   number of qubits per particle
        h_len    (int)   number of qubits for each history step
             
        pReg     (QuantumRegister) particle register
        hReg     (QuantumRegister) history register
        w_hReg   (QuantumRegister) ancillary
        eReg     (QuantumRegister) emission qubit
        wReg     (QuantumRegister) ancillary
        n_aReg   (QuantumRegister) 
        w_aReg   (QuantumRegister) ancillary
        n_bReg   (QuantumRegister)
        w_bReg   (QuantumRegister) ancillary
        n_phiReg (QuantumRegister) 
        w_phiReg (QuantumRegister) ancillary
        circuit  (QuantumCircuit)  that implements QPS
    '''
    def __init__(self, N, ni, g_1, g_2, g_12, eps):
        self._N = N
        self._ni = ni
        self.g_1= g_1
        self.g_2= g_2
        self.g_12= g_12
        self.eps= eps

        # Derived params
        self._L= int(math.ceil(math.log(N + ni, 2)))
        
        gp = math.sqrt(abs((g_1 - g_2) ** 2 + 4 * g_12 ** 2))
        if g_1 > g_2:
            gp = -gp
        self.g_a= (g_1 + g_2 - gp) / 2
        self.g_b= (g_1 + g_2 + gp) / 2
        self.u = math.sqrt(abs((gp + g_1 - g_2) / (2 * gp)))

        # Define some register lengths
        self._p_len = 3
        self._h_len = self._L
        self._e_len = 1
        self._w_len = 5 # All 5 of the work register qubits are used

        #defining the registers
        self.pReg, self.hReg, self.w_hReg, self.eReg, self.wReg, self.n_aReg, self.w_aReg,self.n_bReg, self.w_bReg, \
        self.n_phiReg, self.w_phiReg= self.allocateQubits()

        self._circuit = QuantumCircuit(self.wReg, self.pReg, self.hReg, self.eReg, self.n_phiReg, self.n_aReg,
                                       self.n_bReg, self.w_hReg, self.w_phiReg, self.w_aReg, self.w_bReg)


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
        nqubits_p = self._p_len * (self._N + self._ni)
        nqubits_h = self._N * math.ceil(math.log2((self._N + self._ni)))

        pReg= QuantumRegister(nqubits_p, 'p')
        hReg= QuantumRegister(nqubits_h, 'h')
        w_hReg= QuantumRegister(self._L, 'w_h')
        eReg= QuantumRegister(self._e_len, 'e')
        wReg= QuantumRegister(self._w_len, 'w')
        n_aReg= QuantumRegister(self._L, 'n_a')
        w_aReg= QuantumRegister(self._L, 'w_a')
        n_bReg= QuantumRegister(self._L, 'n_b')
        w_bReg= QuantumRegister(self._L, 'w_b')
        n_phiReg= QuantumRegister(self._L, 'n_phi')
        w_phiReg= QuantumRegister(self._L, 'w_phi')

        return (pReg, hReg, w_hReg, eReg, wReg, n_aReg, w_aReg, n_bReg, w_bReg, n_phiReg, w_phiReg)


    def initializeParticles(self, initialParticles):
        ''' 
        Apply appropriate X gates to ensure that the p register contains all of the initial particles.
            The p registers contains particles in the form of a string '[MSB, middle bit, LSB]'.

        Params:
            initialParticles (List(str)) of the initial particles, where each particle is a binary string (see ptype)    
        '''
        for currentParticleIndex in range(len(initialParticles)):
            for particleBit in range(3):
                pBit= 2 - particleBit # This makes the initial particle strings consistent with the paper convention (also ptype function)
                if int(initialParticles[currentParticleIndex][particleBit]) == 1:
                    self._circuit.x(self.pReg[currentParticleIndex * self._p_len + pBit])


    def flavorControl(self, flavor, control, target, ancilla, control_index, target_index, ancilla_index):
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
        '''
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
            # bosons
            self.flavorControl('phi', self.pReg, self.wReg, self.wReg, (k * self._p_len), 0, 1)
            self.incrementer(l, self.n_phiReg, self.w_phiReg, self.wReg[0])
            self.flavorControl('phi', self.pReg, self.wReg, self.wReg, (k * self._p_len), 0, 1)
            # a fermions
            self.flavorControl('a', self.pReg, self.wReg, self.wReg, (k * self._p_len), 0, 1)
            self.incrementer(l, self.n_aReg, self.w_aReg, self.wReg[0])
            self.flavorControl('a', self.pReg, self.wReg, self.wReg, (k * self._p_len), 0, 1)
            # b fermions
            self.flavorControl('b', self.pReg, self.wReg, self.wReg, (k * self._p_len), 0, 1)
            self.incrementer(l, self.n_bReg, self.w_bReg, self.wReg[0])
            self.flavorControl('b', self.pReg, self.wReg, self.wReg, (k * self._p_len), 0, 1)


    def generateParticleCounts(self, m, k):
        '''
        Return a list with all combinations of n_phi, n_a, and n_b where each n lies in range [0, n_i+m-k],
        and the sum of all n's lies in range [n_i-k, m+n_i-k], all inclusive.

        Params:
            m (int) iteration
            k (int) Uh iteration, see Appendix A.i
        '''
        countsList = []
        for numParticles in range(self._ni - k, m + self._ni - k + 1):
            for numPhi in range(0, self._ni + m - k + 1):
                for numA in range(0, numParticles - numPhi + 1):
                    numB = numParticles - numPhi - numA
                    countsList.append([numPhi, numA, numB])
        return countsList


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

    def numberControl(self, l, number, countReg, workReg):
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
        '''
        if type(number) == int:
            numberBinary = self.intToBinary(l, number)
        else:
            numberBinary = number

        [self._circuit.x(countReg[i]) for i in range(len(numberBinary)) if numberBinary[i] == 0]

        # first level does not use work qubits as control
        if l > 1:
            self._circuit.ccx(countReg[0], countReg[1], workReg[0])
        
        # subfunction to recursively handle toffoli gates
        def binaryToffolis(level):
            self._circuit.ccx(countReg[level], workReg[level - 2], workReg[level - 1])
            if level < l - 1:
                binaryToffolis(level + 1)

        if l > 2:
            binaryToffolis(2)
        # return qubit containing outcome of the operation
        if l == 1:
            return countReg[0]
        else:
            return workReg[l - 2]


    def numberControlT(self, l, number, countReg, workReg):
        ''' CLEANS AFTER numberControl operation. '''
        if type(number) == int:
            numberBinary = self.intToBinary(l, number)
        else:
            numberBinary = number

        # subfunction to recursively handle toffoli gates
        def binaryToffolisT(level):
            if level < l:
                binaryToffolisT(level + 1)
                self._circuit.ccx(countReg[level], workReg[level - 2], workReg[level - 1])

        if l > 2:
            binaryToffolisT(2)
        if l > 1:
            self._circuit.ccx(countReg[0], countReg[1], workReg[0])
        [self._circuit.x(countReg[i]) for i in range(len(numberBinary)) if numberBinary[i] == 0]


    def uE(self, l, m, Delta_phi, Delta_a, Delta_b):
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
        countsList = self.generateParticleCounts(m, 0)

        for counts in countsList:
            n_phi, n_a, n_b = counts[0], counts[1], counts[2]
            Delta = Delta_phi ** n_phi * Delta_a ** n_a * Delta_b ** n_b
            phiControlQub = self.numberControl(l, n_phi, self.n_phiReg, self.w_phiReg)
            aControlQub = self.numberControl(l, n_a, self.n_aReg, self.w_aReg)
            bControlQub = self.numberControl(l, n_b, self.n_bReg, self.w_bReg)
            self._circuit.ccx(phiControlQub, aControlQub, self.wReg[0])
            self._circuit.ccx(bControlQub, self.wReg[0], self.wReg[1])
            self._circuit.cry((2 * math.acos(np.sqrt(Delta))), self.wReg[1], self.eReg[0])
            # undo
            self._circuit.ccx(bControlQub, self.wReg[0], self.wReg[1])
            self._circuit.ccx(phiControlQub, aControlQub, self.wReg[0])
            self.numberControlT(l, n_b, self.n_bReg, self.w_bReg)
            self.numberControlT(l, n_a, self.n_aReg, self.w_aReg)
            self.numberControlT(l, n_phi, self.n_phiReg, self.w_phiReg)


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


    def twoLevelControlledRy(self, l, angle, k, externalControl, reg, workReg):
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
        '''
        grayList = self.generateGrayList(l, k)

        # handle the case where l=0 or 1
        if k == 0:
            return
        if l == 1 and k == 1:
            self._circuit.cry(angle, externalControl, reg[0])
            return

        # swap states according to Gray Code until one step before the end
        for element in grayList:
            targetQub = element[0]
            number = element[1]
            number = number[0:targetQub] + number[targetQub + 1:]
            controlQub = self.numberControl(l - 1, number, reg[0:targetQub] + reg[targetQub + 1:], workReg)
            if element == grayList[-1]:  # reached end
                self._circuit.ccx(controlQub, externalControl, workReg[l - 2])
                self._circuit.cry(angle, workReg[l - 2], reg[targetQub])
                self._circuit.ccx(controlQub, externalControl, workReg[l - 2])
            else:  # swap states
                self._circuit.cx(controlQub, reg[targetQub])
            self.numberControlT(l - 1, number, reg[0:targetQub] + reg[targetQub + 1:], workReg)
        # undo
        for element in self.reverse(grayList[:-1]):
            targetQub = element[0]
            number = element[1]
            number = number[0:targetQub] + number[targetQub + 1:]
            controlQub = self.numberControl(l - 1, number, reg[0:targetQub] + reg[targetQub + 1:], workReg)
            self._circuit.cx(controlQub, reg[targetQub])
            self.numberControlT(l - 1, number, reg[0:targetQub] + reg[targetQub + 1:], workReg)
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


    def U_h(self, l, m, P_phi, P_a, P_b):
        """Implement U_h from paper"""
        for k in range(self._ni + m):
            countsList = self.generateParticleCounts(m, k)  # reduce the available number of particles

            for counts in countsList:
                n_phi, n_a, n_b = counts[0], counts[1], counts[2]

                # controlled R-y from |0> to |k> on all qubits with all possible angles depending on n_phi, n_a, n_b, and flavor
                for flavor in ['phi', 'a', 'b']:
                    angle = self.U_hAngle(flavor, n_phi, n_a, n_b, P_phi, P_a, P_b)

                    #phiControl, aControl, and bControl are the corresponding work registers, and since we call
                    # numberControl we also add x gates on the respective number registers (aka n_phi, n_a, n_b)
                    phiControl = self.numberControl(l, n_phi, self.n_phiReg, self.w_phiReg)
                    # print("qiskit phiControl: ", phiControl)
                    aControl = self.numberControl(l, n_a, self.n_aReg, self.w_aReg)
                    # print("qiskit aControl: ", aControl)
                    bControl = self.numberControl(l, n_b, self.n_bReg, self.w_bReg)
                    # print("qiskit bControl: ", bControl)

                    self._circuit.ccx(phiControl, aControl, self.wReg[0])
                    self._circuit.ccx(bControl, self.wReg[0], self.wReg[1])

                    self.flavorControl(flavor, self.pReg, self.wReg, self.wReg, (k * self._p_len), 2, 4)  # wReg[4] is work qubit but is reset to 0
                    self._circuit.ccx(self.wReg[1], self.wReg[2], self.wReg[3])
                    self._circuit.ccx(self.eReg[0], self.wReg[3], self.wReg[4])

                    self.twoLevelControlledRy(l, angle, k + 1, self.wReg[4], self.hReg[m*self._h_len : (m+1)*self._h_len], self.w_hReg)

                    self._circuit.ccx(self.eReg[0], self.wReg[3], self.wReg[4])  # next steps undo work qubits
                    self._circuit.ccx(self.wReg[1], self.wReg[2], self.wReg[3])
                    self.flavorControl(flavor, self.pReg, self.wReg, self.wReg, (k * self._p_len), 2, 4)  # wReg[4] is work qubit but is reset to 0
                    self._circuit.ccx(bControl, self.wReg[0], self.wReg[1])
                    self._circuit.ccx(phiControl, aControl, self.wReg[0])
                    self.numberControlT(l, n_b, self.n_bReg, self.w_bReg)
                    self.numberControlT(l, n_a, self.n_aReg, self.w_aReg)
                    self.numberControlT(l, n_phi, self.n_phiReg, self.w_phiReg)

            # subtract from the counts register depending on which flavor particle emitted
            for flavor, countReg, workReg in zip(['phi', 'a', 'b'],
                                                 [self.n_phiReg, self.n_aReg, self.n_bReg],
                                                 [self.w_phiReg, self.w_aReg, self.w_bReg]):
                self.flavorControl(flavor, self.pReg, self.wReg, self.wReg, (k * self._p_len), 0, 1)  # wReg[4] is work qubit but is reset to 0
                self.minus1(l, countReg, workReg, self.wReg[0])
                self.flavorControl(flavor, self.pReg, self.wReg, self.wReg, (k * self._p_len), 0, 1)  # wReg[4] is work qubit but is reset to 0

        # apply x on eReg if hReg[m] = 0, apply another x so we essentially control on not 0 instead of 0
        isZeroControl = self.numberControl(l, 0, self.hReg[m*self._L : (m+1)*self._L], self.w_hReg)
        self._circuit.cx(isZeroControl, self.eReg[0])
        self._circuit.x(self.eReg[0])
        self.numberControlT(l, 0, self.hReg[m*self._L : (m+1)*self._L], self.w_hReg)

        # Alternate: use reset
        #circuit.reset(eReg)


    def U_p(self, l, m):
        '''
        Constructs and applies a quantum circuit that implements U_p.

        Params:
            l   (int)   size of the counting & history registers
            m   (int)   iteration
            g_a (float) a-type coupling
            g_b (float) b-type coupling
        '''
        for k in range(0, self._ni + m):
            # Get number control
            controlQub = self.numberControl(l, k + 1, self.hReg[m*self._h_len : (m+1)*self._h_len], self.w_hReg)
            
            # Update particles.
            # first gate in paper U_p
            self._circuit.ccx(controlQub, self.pReg[k * self._p_len + 2], self.pReg[(self._ni + m) * self._p_len + 0])

            # second gate in paper (undoes work register immediately)
            self._circuit.x(self.pReg[k * self._p_len + 1])
            self._circuit.x(self.pReg[k * self._p_len + 2])
            self._circuit.ccx(controlQub, self.pReg[k * self._p_len + 2], self.wReg[0])
            self._circuit.ccx(self.wReg[0], self.pReg[k * self._p_len + 1], self.wReg[1])
            self._circuit.ccx(self.wReg[1], self.pReg[k * self._p_len + 0], self.pReg[(self._ni + m) * self._p_len + 2])
            self._circuit.ccx(self.wReg[0], self.pReg[k * self._p_len + 1], self.wReg[1])
            self._circuit.ccx(controlQub, self.pReg[k * self._p_len + 2], self.wReg[0])
            self._circuit.x(self.pReg[k * self._p_len + 1])
            self._circuit.x(self.pReg[k * self._p_len + 2])

            # third gate in paper
            self._circuit.ccx(controlQub, self.pReg[(self._ni + m) * self._p_len + 2], self.pReg[k * self._p_len + 2])

            # fourth and fifth gate in paper (then undoes work register)
            self._circuit.ccx(controlQub, self.pReg[(self._ni + m) * self._p_len + 2], self.wReg[0])

            # check the format for the control state here
            self._circuit.ch(self.wReg[0], self.pReg[(self._ni + m) * self._p_len + 1])
            angle = (2 * np.arccos(self.g_a / np.sqrt(self.g_a ** 2 + self.g_b ** 2)))
            self._circuit.cry(angle, self.wReg[0], self.pReg[(self._ni + m) * self._p_len + 0])
            self._circuit.ccx(controlQub, self.pReg[(self._ni + m) * self._p_len + 2], self.wReg[0])

            # sixth and seventh gate in paper (then undoes work register)
            self._circuit.x(self.pReg[(self._ni + m) * self._p_len + 0])
            self._circuit.x(self.pReg[(self._ni + m) * self._p_len + 1])
            self._circuit.ccx(self.pReg[(self._ni + m) * self._p_len + 1], self.pReg[(self._ni + m) * self._p_len + 2], self.wReg[0])
            self._circuit.ccx(controlQub, self.wReg[0], self.pReg[k * self._p_len + 1])
            self._circuit.ccx(self.pReg[(self._ni + m) * self._p_len + 1], self.pReg[(self._ni + m) * self._p_len + 2], self.wReg[0])
            self._circuit.ccx(self.pReg[(self._ni + m) * self._p_len + 0], self.pReg[(self._ni + m) * self._p_len + 2], self.wReg[0])
            self._circuit.ccx(controlQub, self.wReg[0], self.pReg[k * self._p_len + 0])
            self._circuit.ccx(self.pReg[(self._ni + m) * self._p_len + 0], self.pReg[(self._ni + m) * self._p_len + 2], self.wReg[0])
            self._circuit.x(self.pReg[(self._ni + m) * self._p_len + 0])
            self._circuit.x(self.pReg[(self._ni + m) * self._p_len + 1])
            
            # Uncompute number control
            self.numberControlT(l, k + 1, self.hReg[m*self._h_len : (m+1)*self._h_len], self.w_hReg)


    def createCircuit(self, initialParticles):
        '''
        This is the main function to create a quantum circuit that implements QPS. 
        The circuit is constructed in place (self._circuit), and also returned.

        Params:
            eps              (float)     scale parameter
            g_1              (float)     type-1 fermion coupling
            g_2              (float)     type-2 fermion coupling
            g_12             (float)     mixed fermion coupling
            initialParticles (List(str)) list of initial particles, each represented by its binary string: 'MSB, middle bit, LSB'
        
        Returns:
            a tuple: QPS circuit (QuantumCircuit), qubit dict (dict)
        '''
        # evaluate P(Theta) and Delta(Theta) at every time step
        self.populateParameterLists()

        qubits = {'pReg': self.pReg, 'hReg': self.hReg, 'w_hReg': self.w_hReg, 'eReg': self.eReg, 'wReg': self.wReg,
                  'n_aReg': self.n_aReg, 'w_aReg': self.w_aReg, 'n_bReg': self.n_bReg, 'w_bReg': self.w_bReg,
                  'n_phiReg': self.n_phiReg, 'w_phiReg': self.w_phiReg}

        self.initializeParticles(initialParticles)

        # begin stepping through subcircuits
        for m in range(self._N):
            l = int(math.floor(math.log(m + self._ni, 2)) + 1)

            # R^(m) - rotate every particle p_k from 1,2 to a,b basis (step 1)
            index = 0
            while index < self.pReg.size:
                self._circuit.cry((2 * math.asin(-self.u)), self.pReg[index + 2], self.pReg[index + 0])
                index += self._p_len

            # populate count register (step 2)
            self.uCount(m, l)

            # assess if emmision occured (step 3)
            self.uE(l, m, self.Delta_phiList[m], self.Delta_aList[m], self.Delta_bList[m])

            # choose a particle to split (step 4)
            self.U_h(l, m, self.P_phiList[m], self.P_aList[m], self.P_bList[m])

            # update particle based on which particle split/emmitted (step 5)
            self.U_p(l, m)

            # R^-(m) rotate every particle p_k from a,b to 1,2 basis (step 6)
            index2 = 0
            while index2 < self.pReg.size:
                # circuit.append(ry(2*math.asin(u)).controlled().on(p_k[2], p_k[0]))
                self._circuit.cry((2 * math.asin(self.u)), self.pReg[index2 + 2], self.pReg[index2 + 0])
                index2 += self._p_len

        print('generated circuit on', self._circuit.num_qubits, 'qubits')
        
        return self._circuit, qubits


    def allocateClbits(self):
        ''' Generate and return all classical registers. '''
        wReg_cl = ClassicalRegister(self._w_len, 'w_cl')
        pReg_cl = []
        for j in range(self._N + self._ni):
            pReg_cl.append(ClassicalRegister(self._p_len, 'p%d_cl' %(j)))
        hReg_cl = []
        for j in range(self._N):
            hReg_cl.append(ClassicalRegister(int(math.ceil(math.log2(self._N + self._ni))), 'h%d_cl' %(j)))
        eReg_cl = ClassicalRegister(self._e_len, 'e_cl')
        n_phiReg_cl = ClassicalRegister(self._L, 'nphi_cl')
        n_aReg_cl = ClassicalRegister(self._L, 'na_cl')
        n_bReg_cl = ClassicalRegister(self._L, 'nb_cl')

        w_hReg_cl = ClassicalRegister(self._L, 'wh_cl')
        w_phiReg_cl = ClassicalRegister(self._L, 'wphi_cl')
        w_aReg_cl = ClassicalRegister(self._L, 'wa_cl')
        w_bReg_cl = ClassicalRegister(self._L, 'wb_cl')

        return (wReg_cl, pReg_cl, hReg_cl, eReg_cl, n_phiReg_cl, n_aReg_cl, n_bReg_cl, 
                w_hReg_cl, w_phiReg_cl, w_aReg_cl, w_bReg_cl)


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
        if type == 'qasm' or type == 'mps':
            if type == 'qasm':
                simulator = Aer.get_backend('qasm_simulator')
            else:
                simulator = qpa.QasmSimulator(method= 'matrix_product_state')

            (wReg_cl, pReg_cl, hReg_cl, # Note: pReg_cl, hReg_cl are lists of ClassicalRegisters
             eReg_cl, n_phiReg_cl, n_aReg_cl, 
             n_bReg_cl, w_hReg_cl, w_phiReg_cl, w_aReg_cl, w_bReg_cl) = self.allocateClbits()
            
            self._circuit.add_register(wReg_cl)
            for j in range(self._N + self._ni):
                self._circuit.add_register(pReg_cl[j])
            for j in range(self._N):
                self._circuit.add_register(hReg_cl[j])
            self._circuit.add_register(eReg_cl)
            self._circuit.add_register(n_phiReg_cl)
            self._circuit.add_register(n_aReg_cl)
            self._circuit.add_register(n_bReg_cl)
            self._circuit.add_register(w_hReg_cl)
            self._circuit.add_register(w_phiReg_cl)
            self._circuit.add_register(w_aReg_cl)
            self._circuit.add_register(w_bReg_cl)

            self._circuit.measure(self.wReg, wReg_cl)
            for j in range(self._N + self._ni):
                self._circuit.measure(self.pReg[3*j : 3*(j+1)], pReg_cl[j])
            for j in range(self._N):
                self._circuit.measure(self.hReg[self._L*j : self._L*(j+1)], hReg_cl[j])
            self._circuit.measure(self.eReg, eReg_cl)
            self._circuit.measure(self.n_phiReg, n_phiReg_cl)
            self._circuit.measure(self.n_aReg, n_aReg_cl)
            self._circuit.measure(self.n_bReg, n_bReg_cl)
            self._circuit.measure(self.w_hReg, w_hReg_cl)
            self._circuit.measure(self.w_phiReg, w_phiReg_cl)
            self._circuit.measure(self.w_aReg, w_aReg_cl)
            self._circuit.measure(self.w_bReg, w_bReg_cl)

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