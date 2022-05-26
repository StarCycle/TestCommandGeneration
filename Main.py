from Parser import Parser
from PQ9Client import PQ9Client
import numpy as np

def GetReward(destID, pq9client, recordCov, covSum):
    # Retrieve coverage array from the target
    succes, rawCov = pq9client.processCommand(destID, [97, 1, 0])
    rawCov = rawCov[5:-2] # Throw away destination, payload size...etc
    binCov = '' # A binary string, where each bit contains a CodeCount result
    binCov = binCov + '0' # The CodeCount ID starts from 1, so the first bit is useless
    for rawData in rawCov:
        binCov = binCov + bin(rawData)[2:].rjust(8, '0')
    reward = 0
    for i in range(len(recordCov)):
        if binCov[i] == '1' and recordCov[i] == 0:
            recordCov[i] = 1
            covSum = covSum + 1
            reward = reward + 1
    return reward, covSum

def RandomPayload(parameters, maxLength):
    payload = []
    payload.append(np.random.choice([3,17,18,19,25])) # service
    payload.append(1)								  # request
    payloadLength = np.random.randint(0, maxLength) 
    payloadParameters = np.random.choice(parameters['cmdParas'], size=payloadLength)
    for parameter in payloadParameters:
        payload.append(np.random.choice(parameters[parameter]['selectFrom']))    
    return payload
	
if __name__ == '__main__':

    # Initialization
    dest = 'COMMS'
    destID = 4
    maxIter = 5000
    maxPayloadLength = 20
    covSum = 0 # initial value
    codeCountNum = 716
    recordCov = [0]*codeCountNum
    parser = Parser('para.csv', 'telem.csv')
    pq9client = PQ9Client('localhost', '10000', 0.5)
    pq9client.connect()
    f = open('curve.txt', 'w')

    # The loop
    for iter in range(maxIter):

        # Ramdomly organize a payload of command
        payload = RandomPayload(parser.parameters[dest], maxPayloadLength)

        # RL takes an action
        succes, rawReply = pq9client.processCommand(destID, payload)
        # reply = parser.ParseReply(rawReply, dest, command['Service'], command['Command'])

        # Calculate Reward
        reward, covSum = GetReward(destID, pq9client, recordCov, covSum)
		
        print(iter, covSum)
        f.write(f'{iter}, {covSum}\n')
