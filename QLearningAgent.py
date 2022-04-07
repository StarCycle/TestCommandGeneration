import numpy as np

class QLearningAgent:

    def __init__(self, actions, learningRate = 0.01, rewardDecay = 0.9, eGreedy = 0.9):
        self.actions = actions
        self.learningRate = learningRate
        self.rewardDecay = rewardDecay
        self.eGreedy = eGreedy
        self.states = [{}] # I am a black state
        self.qTable = np.zeros((1,len(actions))) 

    def ChooseAction(self, state):
        stateID = self.LookForState(state)
        if np.random.uniform() < self.eGreedy:
            stateActionValues = self.qTable[stateID, :]
            # some actions may have the same value, randomly choose on in these actions
            availableActionIDs = np.nonzero(stateActionValues == stateActionValues.max())[0]
            actionID = np.random.choice(availableActionIDs)
        else:
            actionID = np.random.choice(len(self.actions))
        return self.actions[actionID]

    def Learn(self, state, action, reward, nextState):
        stateID = self.LookForState(state)
        nextStateID = self.LookForState(nextState)
        actionID = self.LookForAction(action)
        target = reward + self.rewardDecay * self.qTable[stateID, :].max()
        self.qTable[stateID, actionID] += self.learningRate * (target - self.qTable[stateID, actionID])

    def LookForState(self, state):
        if state not in self.states:
            self.states.append(state)
            self.qTable = np.vstack((self.qTable, np.zeros((1, len(self.actions))) ))
        stateID = self.states.index(state)
        return stateID

    def LookForAction(self, action):
        actionID = self.actions.index(action)
        return actionID
