from PQ9Client import PQ9Client
import csv
from random import randint, choice, sample
import json

def AddParameters(paraFile):
    '''
    Return a dict of parameters. Each parameter has its own dict.
    '''
    parameters = {}
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
            parameters[name]['enumSet'] = {}
            if row[5] != '':
                parameters[name]['enumSet'] = json.loads(row[5])
    return parameters

def GetReward(destID, pq9client, recordCov, covSum):
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

def RandomPayload(parameters, maxLength):
    payload, embedding = [], []
    # Service
    selectedValue = choice(parameters['Service']['selectFrom'])
    embedding.append(parameters['Service']['index'])
    embedding.append(selectedValue)
    payload.append(selectedValue)
    # Request
    embedding.append(parameters['Request']['index'])
    embedding.append(1)
    payload.append(1)
    # Other parameters
    restLength = randint(0, maxLength - 2) 
    otherParameters = sample(list(parameters.keys()), restLength)
    for parameter in otherParameters:
        selectedValue = choice(parameters[parameter]['selectFrom'])
        embedding.append(parameters[parameter]['index'])
        embedding.append(selectedValue)
        payload.append(selectedValue)
    # Padding
    while(len(embedding) < 2*maxLength):
        embedding.append(0)
    return payload, embedding
	
if __name__ == '__main__':

    # Initialization
    dest = 'COMMS'
    destID = 4
    maxIter = 10000
    maxPayloadLength = 20
    covSum = 0 # initial value
    codeCountNum = 714
    recordCov = [0]*(codeCountNum + 1) # The CodeCount ID starts from 1
    embeddings = {'cmd':[], 'cov':[]}
    parameters = AddParameters('para.csv')
    pq9client = PQ9Client('localhost', '10000', 0.5)
    pq9client.connect()

    # The loop
    for iter in range(maxIter):

        # Ramdomly organize a payload of command
        payload, cmdEmbedding = RandomPayload(parameters, maxPayloadLength)
        embeddings['cmd'].append(cmdEmbedding)

        # RL takes an action
        succes, rawReply = pq9client.processCommand(destID, payload)

        # Calculate Reward
        reward, covSum, covEmbedding = GetReward(destID, pq9client, recordCov, covSum)
        embeddings['cov'].append(covEmbedding)
		
        print(iter, covSum)

    with open('dataset.json', 'w') as f:
        json.dump(embeddings, f)
