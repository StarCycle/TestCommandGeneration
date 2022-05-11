from Parser import Parser
from PQ9Client import PQ9Client
from random import randint
from time import time
import json

if __name__ == '__main__':
    parser = Parser('para.csv', 'telec.csv', 'telem.csv')
    allCommands = parser.ListAllCommands('COMMS')

    pq9client = PQ9Client('localhost', '10000', 5)
    pq9client.connect()
	
    cf = open('rawCommands.txt', 'w')
    rf = open('record.txt', 'w')

    startTime = time()

    for i in range(10000):
        # Send command
        cmdNum = len(allCommands) - 1
        cmdID = randint(0, cmdNum)
        command = {}
        command['_send_'] = 'SendRaw'
        command['dest'] = '4' # TODO
        command['src'] = '8'
        command['data'] = ' '.join(str(value) for value in allCommands[cmdID]['rawPayload'])
        cf.write(command['data'] + '\n')
        rf.write(json.dumps(allCommands[cmdID]) + '\n')
        print('Send command: ' + command['data'])
        pq9client.sendFrame(command)
        succes, msg = pq9client.getFrame()
        if succes:
            rawReply = json.loads(msg['_raw_'])
            rawReply = rawReply[3:-2]
            reply = parser.ParseReply(rawReply, 'COMMS', allCommands[cmdID]['Service'], allCommands[cmdID]['Command'])
            rf.write(json.dumps(reply) + '\n')
            print('Command ', i, ' has response at %.2f' % (time()-startTime))
        else:
            break

        # Check status
        command['data'] = '3 1'
        cf.write(command['data'] + '\n')
        pq9client.sendFrame(command)
        succes, msg = pq9client.getFrame()
        if succes:
            rawReply = json.loads(msg['_raw_'])
            rawReply = rawReply[3:-2]
            reply = parser.ParseReply(rawReply, 'COMMS', 'Housekeeping', 'Housekeeping')
            rf.write(json.dumps(reply) + '\n')
            print('Check system status at %.2f' % (time()-startTime))
        else:
            break
