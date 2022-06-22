import csv
import json
from PQ9Client import PQ9Client

def AddParameters(paraFile):
    '''
    Return a dict of parameters. Each parameter has its own dict.
    '''
    parameters, actions = {}, []
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
                if value not in actions:
                    actions.append(value)
            if name == 'Service':
                services = parameters[name]['selectFrom']
            parameters[name]['enumSet'] = {}
            if row[5] != '':
                parameters[name]['enumSet'] = json.loads(row[5])
        actions.append(-1) # -1 means sending the command now
    return parameters, actions, services

def CheckCov(destID, pq9client, recordCov):
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
            reward = reward + 1
    covByLastCmd = [int(value) for value in list(binCov)]
    return reward, covByLastCmd
	
class MyEnv():

    def __init__(self, destID, paraFile, maxSteps, maxPayloadLength, codeCountNum):
        self.destID = destID
        self.maxSteps = maxSteps
        self.maxPayloadLength = maxPayloadLength
        self.codeCountNum = codeCountNum
        self.paras, self.actions, self.services  = AddParameters(paraFile)
        self.state = [0]*(codeCountNum + maxPayloadLength)
        self.stateMin = [0]*(codeCountNum + maxPayloadLength)
        self.stateMax = [1]*codeCountNum + [len(self.services) - 1] + [1] + [len(self.actions) - 1]*(maxPayloadLength-2)
        self.cmd = []
        self.recordCov = [0]*(self.codeCountNum + 1) # The CodeCount ID starts from 1
        self.pq9client = PQ9Client('localhost', '10000', 0.5)
        self.pq9client.connect()

    def normalizedState(self):
        outputState = []
        for i in range(len(self.state)):
            outputState.append((self.state[i] - self.stateMin[i]) / (self.stateMax[i] - self.stateMin[i]))
        return outputState

    def reset(self):
        # Reset internal paramaters
        self.steps = 0
        self.state = [0]*len(self.state)
        self.cmd = []
        self.recordCov = [0]*len(self.recordCov)
        self.file = open('cmds.txt', 'w')
        # Reset the target board
        succes, rawReply = self.pq9client.processCommand(self.destID, [19, 1])
        reward, covByLastCmd = CheckCov(self.destID, self.pq9client, self.recordCov)
        self.state[:self.codeCountNum] = self.recordCov[1:]
        return self.normalizedState()

    def step(self, actionID):
        reward = 0
        self.steps += 1
        if self.cmd == []: # Select service
            serviceID = actionID % len(self.services)
            self.state[self.codeCountNum] = serviceID
            self.state[self.codeCountNum+1] = 1
            self.cmd.append(self.services[serviceID])
            self.cmd.append(1)
        elif self.actions[actionID] == -1: # Send the command now
            succes, rawReply = self.pq9client.processCommand(self.destID, self.cmd)
            self.file.write(' '.join(str(value) for value in self.cmd) + '\n')
            reward, covByLastCmd = CheckCov(self.destID, self.pq9client, self.recordCov)
            self.state = [0]*len(self.state)
            self.state[:self.codeCountNum] = self.recordCov[1:]
            self.cmd = []
        else:
            self.state[self.codeCountNum + len(self.cmd)] = actionID
            self.cmd.append(self.actions[actionID])
            if len(self.cmd) >= self.maxPayloadLength:
                succes, rawReply = self.pq9client.processCommand(self.destID, self.cmd)
                self.file.write(' '.join(str(value) for value in self.cmd) + '\n')
                reward, covByLastCmd = CheckCov(self.destID, self.pq9client, self.recordCov)
                self.state = [0]*len(self.state)
                self.state[:self.codeCountNum] = self.recordCov[1:]
                self.cmd = []
        if self.steps > self.maxSteps:
            return self.normalizedState(), reward, 1, {}
        else:
            return self.normalizedState(), reward, 0, {}

