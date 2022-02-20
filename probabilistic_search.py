# An Incomplete Implementation of 
# Discovering neural nets with low Kolmogorov complexity and high generalization capability
# Schmidhuber, J, Neural Networks, 1997
import numpy as np
from numpy.random import MT19937
from sklearn.utils import shuffle

from numpy.random import RandomState, SeedSequence

rs = RandomState(MT19937(SeedSequence(123456789)))

# Program Tape address space in [0, sp]
# Work    Tape address space in [-sw, -1]
# -sw   -sw+1   ....    -2 -1 0 1 2 ... sp-1   sp
#[----- work tape ----------] [--- program tape -]

ERR_EXCEED_SP = 1
ERR_STOP = 2 # graceful stop
ERR_EMPTYWORKTAPE = 3
ERR_HALT = 4 # premature stop
ERR_NO_VALID_ADDRESS_FOUND = 5

INS_JUMPLEQ = 0
INS_WRITEWEIGHT = 1
INS_JUMP = 2
INS_STOP = 3
INS_ADD = 4
INS_GETINPUT = 5
INS_MOVE = 6
INS_ALLOCATE = 7
INS_INCREMENT = 8
INS_DECREMENT = 9
INS_SUBTRACT = 10
INS_MULTIPLY = 11
INS_FREE = 12

instructionNames = {INS_JUMPLEQ: 'JUMPLEQ', 
                    INS_WRITEWEIGHT: 'WRITEW',
                    INS_JUMP: 'JUMP',
                    INS_STOP: 'STOP',
                    INS_ADD: 'ADD',
                    INS_GETINPUT: 'GETINPUT',
                    INS_MOVE: 'MOVE',
                    INS_ALLOCATE: 'ALLOCATE',
                    INS_INCREMENT: 'INCREMENT',
                    INS_DECREMENT: 'DECREMENT',
                    INS_SUBTRACT: 'SUBTRACT',
                    INS_MULTIPLY: 'MULTIPLY',
                    INS_FREE: 'FREE' }

class Program:
    
    def __init__(self):
        self.sw = 100
        self.sp = 100
        self.maxint = 10000
        self.pt = np.zeros(self.sp+1,dtype=np.int32)
        self.wt = np.zeros(self.sw+1,dtype=np.int32)
        self.weights = np.zeros(100,dtype=np.int32)
        self.ni = 20
        self.inputs = np.zeros(self.ni,dtype=np.int32)

        self.reset()

    def reset(self):
        for i in range(self.sp + 1):
            self.pt[i] = 0
        for i in range(self.sw + 1):
            self.wt[i] = 0
        for i in range(self.ni):
            self.inputs[i] = 0

        # Initialization
        self.OracleAddress = 0
        self.InstructionPointer = 0
        self.WeightPointer = 0
        self.Min = 0
        self.Max = -1
        self.CurrentRuntime = 0


    def __str__(self):
        out = ''
        for addr in range(self.Min, self.Max + 1):
            out = out + '(%d)%d,'%(addr, self.readValue(addr)) 
        return out

    def __repr__(self):
        out = ''
        for addr in range(self.Min, self.Max + 1):
            out = out + '(%d)%d,'%(addr, self.readValue(addr)) 
        return out

    def chooseRandom(self, a, b):
        return a + np.random.randint(b-a+1)

    def readValue(self, address):
        if address >= 0 and address <= self.Max:
            return self.pt[address]
        elif address < 0 and  address >= self.Min:
            return self.wt[-address-1]
        else:
            return np.nan

    def writeValue(self, address, value):
        if address < 0 and  address >= self.Min:
            self.wt[-address-1] = value
            return True
        else:
            return False

    def execute(self):
        insptr = self.InstructionPointer
        ins = self.pt[insptr]
        if ins == INS_JUMPLEQ:
            addr1 = self.pt[insptr+1]
            addr2 = self.pt[insptr+2]
            addr3 = self.pt[insptr+3]
            val1 = self.readValue(addr1)
            val2 = self.readValue(addr2)
            if val1 <= val2:
                self.InstructionPointer = addr3
        elif ins == INS_WRITEWEIGHT:
            addr1 = self.pt[insptr+1]
            if self.WeightPointer >= 100:
                return False
            val1 = self.readValue(addr1)
            if np.isnan(val1):
                return False
            self.weights[self.WeightPointer] = val1
            self.WeightPointer += 1
            self.InstructionPointer = insptr + 2
        elif ins == INS_JUMP:
            addr1 = self.pt[insptr + 1]
            self.InstructionPointer = addr1 
        elif ins == INS_STOP:
            return False
        elif ins == INS_ADD:
            addr1 = self.pt[insptr+1]
            addr2 = self.pt[insptr+2]
            addr3 = self.pt[insptr+3]
            val1 = self.readValue(addr1)
            val2 = self.readValue(addr2)
            if np.isnan(val1) or np.isnan(val2) or self.writeValue(addr3, val1 + val2) == False:
                return False
            self.InstructionPointer = insptr + 4
        elif ins == INS_GETINPUT:
            addr1 = self.pt[insptr+1]
            addr2 = self.pt[insptr+2]
            i = self.readValue(addr1)
            if np.isnan(i):
                return False
            if i < 0 or i >= self.ni:
                return False
            if self.writeValue(addr2, self.inputs[i]) == False:
                return False     
            self.InstructionPointer = insptr + 3
        elif ins == INS_MOVE:
            addr1 = self.pt[insptr+1]
            addr2 = self.pt[insptr+2]
            val1 = self.readValue(addr1)
            if np.isnan(val1):
                return False
            if self.writeValue(addr2, val1) == False:
                return False
            self.InstructionPointer = insptr + 3
        elif ins == INS_ALLOCATE:
            addr1 = self.pt[insptr+1]
            ncells = self.readValue(addr1)
            if np.isnan(ncells):
                return False
            if self.Min - ncells < - self.sw:
                return False
            for i in range(ncells):
                # logical index = self.Min - i - 1
                # real index = -address - 1 = i - self.Min
                self.wt[i - self.Min] = 0
            self.Min = self.Min - ncells
            self.InstructionPointer = insptr + 2 
        elif ins == INS_INCREMENT:
            addr1 = self.pt[insptr+1]
            val1 = self.readValue(addr1)
            if np.isnan(val1):
                return False
            if self.writeValue(addr1, val1 + 1) == False:
                return False

            self.InstructionPointer = insptr + 2 
        elif ins == INS_DECREMENT:
            addr1 = self.pt[insptr+1]
            val1 = self.readValue(addr1)
            if np.isnan(val1):
                return False
            if self.writeValue(addr1, val1 - 1) == False:
                return False
            self.InstructionPointer = insptr + 2            
        elif ins == INS_SUBTRACT:
            addr1 = self.pt[insptr+1]
            addr2 = self.pt[insptr+2]
            addr3 = self.pt[insptr+3]
            val1 = self.readValue(addr1)
            val2 = self.readValue(addr2)
            if np.isnan(val1) or np.isnan(val2) or self.writeValue(addr3, val2 - val1) == False:
                return False
            self.InstructionPointer = insptr + 4
        elif ins == INS_MULTIPLY:
            addr1 = self.pt[insptr+1]
            addr2 = self.pt[insptr+2]
            addr3 = self.pt[insptr+3]
            val1 = self.readValue(addr1)
            val2 = self.readValue(addr2)
            if np.isnan(val1) or np.isnan(val2) or self.writeValue(addr3, val1 * val2) == False:
                return False
            self.InstructionPointer = insptr + 4
        elif ins == INS_FREE:
            addr1 = self.pt[insptr+1]
            ncells = self.readValue(addr1)
            if np.isnan(ncells):
                return False
            if self.Min + ncells > -1:
                return False
            self.Min = self.Min + ncells            
        return True
            
    def addInstruction(self):
        primitiveIndex = self.chooseRandom(0,12)
        if self.OracleAddress >= self.sp and primitiveIndex != INS_STOP:
            return ERR_EXCEED_SP
        elif self.OracleAddress > self.sp and primitiveIndex == INS_STOP:
            return ERR_EXCEED_SP
        if primitiveIndex == INS_JUMPLEQ:
            if self.OracleAddress+3 > self.sp:
                return ERR_EXCEED_SP
            if self.Min > self.OracleAddress + 3:
                return ERR_EXCEED_SP
            self.pt[self.OracleAddress] = primitiveIndex
            legaladdress1 = self.chooseRandom(self.Min,self.OracleAddress + 3)
            legaladdress2 = self.chooseRandom(self.Min,self.OracleAddress + 3)
            legaladdress3 = self.chooseRandom(self.Min,self.OracleAddress + 4)
            self.pt[self.OracleAddress+1] = legaladdress1
            self.pt[self.OracleAddress+2] = legaladdress2
            self.pt[self.OracleAddress+3] = legaladdress3
            self.Max = self.OracleAddress + 3
            self.OracleAddress = self.Max + 1 
        elif primitiveIndex == INS_ADD \
            or primitiveIndex == INS_SUBTRACT \
            or primitiveIndex == INS_MULTIPLY:
            if self.OracleAddress+3 > self.sp:
                return ERR_EXCEED_SP
            if self.Min > -1 :
                return ERR_EMPTYWORKTAPE
            if self.Min > self.OracleAddress + 3:
                return ERR_EXCEED_SP

            self.pt[self.OracleAddress] = primitiveIndex
            legaladdress1 = self.chooseRandom(self.Min,self.OracleAddress + 3)
            legaladdress2 = self.chooseRandom(self.Min,self.OracleAddress + 3)
            legaladdress3 = self.chooseRandom(self.Min,-1)
            self.pt[self.OracleAddress+1] = legaladdress1
            self.pt[self.OracleAddress+2] = legaladdress2
            self.pt[self.OracleAddress+3] = legaladdress3
            self.Max = self.OracleAddress + 3
            self.OracleAddress = self.Max + 1
        elif primitiveIndex == INS_WRITEWEIGHT \
            or primitiveIndex == INS_ALLOCATE \
            or primitiveIndex == INS_FREE:
            if self.Min > self.OracleAddress + 1:
                return ERR_NO_VALID_ADDRESS_FOUND
            if self.Min > self.OracleAddress + 1:
                return ERR_EXCEED_SP
            legaladdress1 = self.chooseRandom(self.Min, self.OracleAddress + 1)    
            self.pt[self.OracleAddress] = primitiveIndex
            self.pt[self.OracleAddress + 1] = legaladdress1
            self.Max = self.OracleAddress + 1
            self.OracleAddress = self.Max + 1
        elif primitiveIndex == INS_JUMP:
            if self.OracleAddress + 1 > self.sp:
                return ERR_EXCEED_SP
            self.pt[self.OracleAddress] = primitiveIndex
            if self.Min > self.OracleAddress + 1:
                return ERR_EXCEED_SP
            self.pt[self.OracleAddress + 1] = self.chooseRandom(self.Min, self.OracleAddress + 1)
            self.Max = self.OracleAddress + 1
            self.OracleAddress = self.Max + 1
        elif primitiveIndex == INS_INCREMENT or primitiveIndex == INS_DECREMENT:
            if self.OracleAddress + 1 > self.sp:
                return ERR_EXCEED_SP
            if self.Min > -1 :
                return ERR_NO_VALID_ADDRESS_FOUND
            self.pt[self.OracleAddress] = primitiveIndex
            self.pt[self.OracleAddress + 1] = self.chooseRandom(self.Min, -1)
            self.Max = self.OracleAddress + 1
            self.OracleAddress = self.Max + 1            
        elif primitiveIndex == INS_STOP:
            self.pt[self.OracleAddress] = primitiveIndex
            self.Max = self.OracleAddress
            self.OracleAddress = self.Max + 1
        elif primitiveIndex == INS_GETINPUT or primitiveIndex == INS_MOVE:
            if self.OracleAddress + 2 > self.sp:
                return ERR_EXCEED_SP
            if self.Min > -1 :
                return ERR_EMPTYWORKTAPE
            if self.Min > self.OracleAddress + 2:
                return ERR_EXCEED_SP
            self.pt[self.OracleAddress] = primitiveIndex
            self.pt[self.OracleAddress + 1] = self.chooseRandom(self.Min, self.OracleAddress + 2 )    
            self.pt[self.OracleAddress + 2] = self.chooseRandom(self.Min, -1)   
            self.Max = self.OracleAddress + 2
            self.OracleAddress = self.Max + 1 
        return 0

    def cycle(self):
        if self.InstructionPointer == self.OracleAddress:
            if self.addInstruction():
                return False
        return self.execute()
    
    def check(self):
        return all(self.weights == np.ones(100, dtype=np.int32))

    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, Program):
            if self.Max == other.Max and self.Max >= 0:
                return all(self.pt[:self.Max] == other.pt[:other.Max])
        return False

all_x = np.zeros((161700,100))
all_y = 3*np.ones(161700)
count = 0
for i in range(100):
    for j in range(i+1,100):
        for k in range(j+1,100):
            all_x[count,[i,j,k]] = 1
            count += 1

all_x = shuffle(all_x)
trn_x = all_x[:3]
trn_y = all_y[:3]
tst_x = all_x[3:]
tst_y = all_y[3:]

results = []
for trial in range(10000):
    p = Program()
    cycle = 0
    while p.cycle() and cycle < 1000:
        cycle += 1
    trn_perf = all(trn_y == np.dot(trn_x,p.weights))
    tst_perf = all(tst_y == np.dot(tst_x,p.weights))
    if trn_perf and p not in results:
        results.append(p)
        print({'trial': trial, 'cycle': cycle, 'training':trn_perf, 'testing':tst_perf, 'program':p})
