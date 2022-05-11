from Parser import Parser
from pprint import pprint
parser = Parser('para.csv', 'telec.csv', 'telem.csv')
allCommands = parser.ListAllCommands('COMMS')
pprint(allCommands)
