from QLearningAgent import QLearningAgent
from StateIdentification import StateIdentification
from Parser import Parser
from PQ9Client import PQ9Client
import networkx as nx
import matplotlib.pyplot as plt

def GetReward(destID, pq9client, coverage):
    # Retrieve coverage array from the target
    succes, rawCov = pq9client.processCommand(destID, [97, 1, 0])
    rawCov = rawCov[5:-2] # Throw away destination, payload size...etc
    binCov = '' # A binary string, where each bit contains a CodeCount result
    binCov = binCov + '0' # The CodeCount ID starts from 1, so the first bit is useless
    for rawData in rawCov:
        binCov = binCov + bin(rawData)[2:].rjust(8, '0')
    reward = binCov.count('1') - coverage
    coverage = binCov.count('1')
    return reward, coverage
	
def UpdateControlFlow(destID, pq9client, nextNodes, lastNodes, codeCountNum):
    binData = ''
    newEdgeNum = 0
    for i in range(12): # 12 regions of the transition record
        succes, rawData = pq9client.processCommand(destID, [97, 1, 1, i])
        for rawValue in rawData[5:-2]: # Throw away destination, payload size...etc
            binData = binData + bin(rawValue)[2:].rjust(8, '0')
    for startNode in range(1, codeCountNum):
        binValue = binData[(startNode*16+8):(startNode+1)*16] + binData[startNode*16:(startNode*16+8)] # Switch the order
        endNode = int(binValue, 2)
        if endNode is not 0 and endNode not in nextNodes[startNode]:
            nextNodes[startNode].append(endNode)
            lastNodes[endNode].append(startNode)
            newEdgeNum = newEdgeNum + 1
    # print(newEdgeNum, ' new edges in the control flow')

if __name__ == '__main__':

    # Initialization
    dest = 'COMMS'
    destID = 4
    maxIter = 5000
    coverage = 0 # initial value
    state = {}	 # initial value
    takeControlFlowStep = 100
    codeCountNum = 716
    nextNodes = [[] for i in range(codeCountNum+1)] # The first element is useless
    lastNodes = [[] for i in range(codeCountNum+1)]
    parser = Parser('para.csv', 'telec.csv', 'telem.csv')
    allCommands = parser.ListAllCommands(dest)
    stateIdt = StateIdentification()
    agent = QLearningAgent(allCommands, eGreedy = 0)
    pq9client = PQ9Client('localhost', '10000', 5)
    pq9client.connect()

    # The loop
    for iter in range(maxIter):

        # RL chooses an action
        command = agent.ChooseAction(state)

        # RL takes an action
        succes, rawReply = pq9client.processCommand(destID, command['rawPayload'])
        reply = parser.ParseReply(rawReply, dest, command['Service'], command['Command'])
        newState = stateIdt.NewState(state, command, reply) 

        # Calculate Reward
        reward, coverage = GetReward(destID, pq9client, coverage)
		
        if iter % takeControlFlowStep == 0:
            UpdateControlFlow(destID, pq9client, nextNodes, lastNodes, codeCountNum)

        # RL learns from this transition
        agent.Learn(state, command, reward, newState)

        # Swap states
        state = newState
        print(iter, len(agent.states), coverage)

    controlFlowGraph = nx.DiGraph()
    for startNode in range(len(nextNodes)):
        if nextNodes[startNode] is not []:
            edges = zip( [startNode]*len(nextNodes[startNode]), nextNodes[startNode] )
            controlFlowGraph.add_edges_from(list(edges))
    nx.draw_kamada_kawai(controlFlowGraph, with_labels=False, node_size=10, arrows=True, arrowsize=5) 
    plt.show()
