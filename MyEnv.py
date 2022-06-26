from Parser import Parser
from PQ9Client import PQ9Client
from random import randint

def CkeckCov(destID, pq9client, recordCov, covSum):
    # Retrieve coverage array from the target
    succes, rawCov = pq9client.processCommand(destID, [97, 1, 0])
    rawCov = rawCov[5:-2] # Throw away destination, payload size...etc
    binCov = '' # A binary string, where each bit contains a CodeCount result
    binCov = binCov + '0' # The CodeCount ID starts from 1, so the first bit is useless
    for rawData in rawCov:
        binCov = binCov + bin(rawData)[2:].rjust(8, '0')
    binCov = binCov[:len(recordCov)]
    reward = 0
    for i in range(len(recordCov)):
        if binCov[i] == '1' and recordCov[i] == 0:
            recordCov[i] = 1
            covSum = covSum + 1
            reward = reward + 1
    return reward, covSum, [int(value) for value in list(binCov)]

class MyEnv():

    def __init__(self, dest, destID, paraFile, cmdFile, replyFile, codeCountNum):
        self.destID = destID
        self.parser = Parser(paraFile, cmdFile, replyFile)
        self.codeCountNum = codeCountNum
        self.covSum = 0 # Initial value
        self.recordCov = [0]*(self.codeCountNum + 1) # The CodeCount ID starts from 1
        self.state = self.recordCov
        self.actions = self.parser.ListAllCommands(dest)
        self.pq9client = PQ9Client('localhost', '10000', 0.5)
        self.pq9client.connect()

    def reset(self):
        # Reset internal paramaters
        self.covSum = 0
        self.recordCov = [0]*len(self.recordCov)
        # Reset the target board
        succes, rawReply = pq9client.processCommand(self.destID, [19, 1])
        reward, self.covSum, covByLastCmd = CheckCov(self.destID, self.pq9client, self.recordCov, self.covSum)
        return self.state

    def step(self, cmdID):
        reward = 0
        succes, rawReply = pq9client.processCommand(self.destID, allCommands[cmdID]['rawPayload'])
        reward, self.covSum, covByLastCmd = CheckCov(self.destID, self.pq9client, self.recordCov, self.covSum)
        return self.state, reward, False, {}

