from Parser import Parser
from PQ9Client import PQ9Client
from random import randint
from time import sleep

def CheckCov(destID, pq9client, recordCov, covSum):
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

    def __init__(self, dest, destID, paraFile, cmdFile, replyFile, num_epoch_steps, codeCountNum):
        self.destID = destID
        self.parser = Parser(paraFile, cmdFile, replyFile)
        self.codeCountNum = codeCountNum
        self.covSum = 0 # Initial value
        self.recordCov = [0]*(self.codeCountNum + 1) # The CodeCount ID starts from 1
        self.actions = self.parser.ListAllCommands(dest)
        self.max_num_steps = num_epoch_steps
        self.current_step = 0
        self.pq9client = PQ9Client('localhost', '10000', 0.5)
        self.pq9client.connect()

    def reset(self):
        # Reset internal paramaters
        self.covSum = 0
        self.current_step = 0
        self.recordCov = [0]*len(self.recordCov)
        # Reset the target board
        succes, rawReply = self.pq9client.processCommand(self.destID, [19, 1, 1])
        reward, self.covSum, covByLastCmd = CheckCov(self.destID, self.pq9client, self.recordCov, self.covSum)
        return self.recordCov

    def step(self, actionID):
        reward = 0
        done = False
        succes, rawReply = self.pq9client.processCommand(self.destID, self.actions[actionID]['rawPayload'])
        reward, self.covSum, covByLastCmd = CheckCov(self.destID, self.pq9client, self.recordCov, self.covSum)
        self.current_step += 1
        if self.current_step >= self.max_num_steps:
            done = True
        return self.recordCov, reward, done, {}

    def randomBaseline(self):
        file = open('curve.txt', 'w')
        for i in range(5):
            cumuReward = 0
            self.reset()
            for j in range(self.max_num_steps):
                actionID = randint(0, len(self.actions) - 1)
                recordCov, reward, done, info = self.step(actionID)
                cumuReward += reward
                print(j, cumuReward, self.covSum)
                file.write(str(j) + ',' + str(cumuReward) + ',' + str(self.covSum) + '\n')

    def fixedBaseline(self):
        file = open('curve.txt', 'w')
        for i in range(50):
            cumuReward = 0
            if self.covSum > 235:
                print('here')
            self.reset()
            index = 0
            for k in range(5):
                for j in range(len(self.actions)):
                    actionID = j
                    recordCov, reward, done, info = self.step(actionID)
                    cumuReward += reward
                    print(index, cumuReward, self.covSum)
                    file.write(str(index) + ',' + str(cumuReward) + ',' + str(self.covSum) + '\n')
                    index = index+1

