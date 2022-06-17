from PQ9Client import PQ9Client
import csv
from random import randint, choice, sample
import json

def AddParameters(paraFile):
    '''
    Return a dict of parameters. Each parameter has its own dict.
    '''
    parameters, actions = {}, [['SendCmdNow', 0, 0]]
    with open(paraFile, encoding = 'utf-8', errors = 'replace') as csvFile:
        csvReader = csv.reader(csvFile)
        index = -2
        for row in csvReader:
            index = index + 1
            if index == -1: # First line of paraFile
                firstLine = False
                continue
            name = row[1]
            parameters[name] = {}
            parameters[name]['index'] = index   # Start from 0
            parameters[name]['subsystem'] = row[0]
            parameters[name]['type'] = row[2]
            parameters[name]['size'] = int(row[3])
            parameters[name]['selectFrom'] = json.loads(row[4]) 
            for value in parameters[name]['selectFrom']:
                actions.append([name, index, value])
            parameters[name]['enumSet'] = {}
            if row[5] != '':
                parameters[name]['enumSet'] = json.loads(row[5])
    return parameters, actions

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

    def __init__(self, destID, paraFile, maxPayloadLength, codeCountNum):
        self.destID = destID
        self.maxPayloadLength = self.maxPayloadLength
        self.codeCountNum = codeCountNum
        self.covSum = 0 # Initial value
        self.state = [0]*(codeCountNum + maxPayloadLength*2)
        self.cmd = []
        self.recordCov = [0]*(self.codeCountNum + 1) # The CodeCount ID starts from 1
        self.paras, self.actions = AddParameters(paraFile)
        self.pq9client = PQ9Client('localhost', '10000', 0.5)
        self.pq9client.connect()

    def reset(self):
        # Reset internal paramaters
        self.covSum = 0
        self.state = [0]*len(self.state)
        self.cmd = []
        self.recordCov = [0]*len(self.recordCov)
        # Reset the target board
        succes, rawReply = pq9client.processCommand(self.destID, [19, 1])
        reward, self.covSum, covByLastCmd = CheckCov(self.destID, self.pq9client, self.recordCov, self.covSum)
        self.state[:codeCountNum] = self.recordCov[1:]
        self.justReset = True
        return self.state

    def step(self, actionID):
        reward = 0
        if self.justReset == True:
            if self.actions[actionID][0] == 'Service':
                self.state[codeCountNum] = self.paras['Service']['index']
                self.state[codeCountNum+1] = self.actions[actionID][1]
                self.state[codeCountNum+2] = self.paras['Request']['index']
                self.state[codeCountNum+3] = 1
                self.payloadLength = 2
                self.justReset == False
        elif self.actions[actionID][0] == 'SendCmdNow':
            succes, rawReply = pq9client.processCommand(self.destID, cmd)
            reward, self.covSum, covByLastCmd = CheckCov(self.destID, self.pq9client, self.recordCov, self.covSum)
            self.state[:codeCountNum] = self.recordCov[1:]
            self.state[codeCountNum:] = 0
            self.cmd = []
        else:
            self.state[codeCountNum + 2*self.payloadLength] = self.actions[actionID][1]
            self.state[codeCountNum + 2*self.payloadLength + 1] = self.actions[actionID][2]
            self.cmd.append(self.actions[actionID][2])
            if len(self.cmd) >= self.maxPayloadLength:
                succes, rawReply = pq9client.processCommand(self.destID, cmd)
                reward, self.covSum, covByLastCmd = CheckCov(self.destID, self.pq9client, self.recordCov, self.covSum)
                self.state[:codeCountNum] = self.recordCov[1:]
                self.state[codeCountNum:] = 0
                self.cmd = []
        return self.state, reward, False, {}

