import csv
import json
from random import randint
from Parser import Parser
from PQ9Client import PQ9Client
from time import time, sleep

def AddParameters(paraFile):
    '''
    Return a list of values of parameters
    '''
    parameters = []
    with open(paraFile, encoding = 'utf-8', errors = 'replace') as csvFile:
        csvReader = csv.reader(csvFile)
        index = -2
        for row in csvReader:
            index = index + 1
            if index == -1: # First line of paraFile
                firstLine = False
                continue
            if row[8] != 'Reply Only':
                for value in json.loads(row[8]):
                    if value not in parameters:
                        parameters.append(value)
    return parameters

def CheckLoopCount(destID, pq9client):
    succes, rawLoopCount = pq9client.processCommand(destID, [97, 1, 1])
    rawLoopCount = rawLoopCount[5:9]
    binLoopCount = ''
    for rawData in rawLoopCount:
        binLoopCount = binLoopCount + bin(rawData)[2:].rjust(8, '0')
    loopCount = int(binLoopCount, 2)
    return loopCount

def CheckCov(destID, pq9client, recordCov, covSum):
    # Retrieve coverage array from the target
    succes, rawCov = pq9client.processCommand(destID, [97, 1, 0])
    if succes == False:
        return False, 0, covSum, []
    rawCov = rawCov[5:-2] # Throw away destination, payload size...etc
    binCov = '' # A binary string, where each bit contains a CodeCount result
    binCov = binCov + '0' # The CodeCount ID starts from 1, so the first bit is useless
    for rawData in rawCov:
        binCov = binCov + bin(rawData)[2:].rjust(8, '0')
    if len(binCov) < len(recordCov):
        return 0, covSum, []
    reward = 0
    for i in range(len(recordCov)):
        if binCov[i] == '1' and recordCov[i] == 0:
            recordCov[i] = 1
            covSum = covSum + 1
            reward = reward + 1
    return reward, covSum, [int(value) for value in list(binCov)]

class MyEnv():

    def __init__(self, dest, destID, paraFile, services, max_payload_length, num_epoch_steps, codeCountNum):
        self.destID = destID
        self.max_payload_length = max_payload_length
        self.paras = AddParameters(paraFile)
        self.services = services
        self.codeCountNum = codeCountNum
        self.covSum = 0 # Initial value
        self.recordCov = [0]*(self.codeCountNum + 1) # The CodeCount ID starts from 1
        self.action_space = [len(services)] + [len(self.paras) + 1]*max_payload_length # including no_op
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
        self.pq9client.reset(self.destID)
        reward, self.covSum, covByLastCmd = CheckCov(self.destID, self.pq9client, self.recordCov, self.covSum)
        return self.recordCov

    def step(self, action):
        reward = 0
        done = False
        cmd = [self.services[action[0]], 1] # Service and Request
        for actionID in action[1:]:         # Payload of the commands
            if actionID != len(self.paras): # Null operation
                cmd.append(self.paras[actionID])
        startTime = time()
        _ = CheckLoopCount(self.destID, self.pq9client)
        succes, rawReply = self.pq9client.processCommand(self.destID, cmd)         
        newCov, self.covSum, covByLastCmd = CheckCov(self.destID, self.pq9client, self.recordCov, self.covSum)
        endTime = time()
        loopCount = CheckLoopCount(self.destID, self.pq9client)
        reward = 48000 / (loopCount/(endTime - startTime)) # How many clock period are used in a loop (unit: 1k)
        self.current_step += 1
        if self.current_step >= self.max_num_steps or succes == False:
            done = True
        return self.recordCov, reward, done, {}

    def randomBaseline(self):
        self.reset()
        for step in range(self.max_num_steps):
            action = []
            for value_range in self.action_space:
                action.append(randint(0, value_range - 1))
            _, reward, done, info = self.step(action)
            print(step, self.covSum)