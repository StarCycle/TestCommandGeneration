import csv
import copy
import json

class Parser:

    def __init__(self, paraFile, replyFile):
        self.AddParameters(paraFile)
        self.AddReplies(replyFile)
        return

    def ParseReply(self, rawReply, source, service, command):
        '''
        Translate the received decimal list to a dict of {parameter_name: parameter_value}
        Input:
            rawReply    a list like [1, 2, 8, 17, 2]
            source      a string shows where this rawFrame comes from, like "COMMS"
            service     a string shows which service the rawFrame belongs to, like "Ping"
            command     a string representing the command, like "Ping"
        Output:
            trFrame     translated dict of rawFrame
        '''
        rawFrame = rawReply[3:-2]
        trFrame = {'Source':source, 'Service':service, 'Command':command, 'embedding':[0]*self.parameterNum}
        index = 0 # This function has read how many bits from the rawFrame
        reply = self.replies[source][service][command]
        for parameter in reply:
            paraDefn = self.parameters[source][parameter]
            size = paraDefn['size']
            startByte = index // 8
            if startByte > len(rawFrame) - 1: # early stop (usually means failure of command)
                break;
            if size == 'unknown':
                # if the payload is special data, we don't translate it
                trFrame[parameter] = rawFrame[startByte:]
                return trFrame
            elif size == 1:
                endByte = (index+size) // 8
                binData = bin(rawFrame[startByte])[2:].rjust(8, '0')[index%8]
                trFrame[parameter] = self.ParseData(binData, paraDefn)
                index += 1
            else:
                endByte = (index+size) // 8
                decList = rawFrame[startByte:endByte]
                binData = ''
                for dec in decList:
                    binData = binData + bin(dec)[2:].rjust(8, '0')
                trFrame[parameter] = self.ParseData(binData[0:size], paraDefn)
                index += size
        return trFrame

    def ParseData(self, binData, paraDefn):
        '''
        Retrieve a parameter from a binary string
        Input:
            rawData     A binary string of a parameter
            paraDefn    Definition of the parameter
        Output:
            value       Translated value from the binary string
        '''
        if paraDefn['encoding'] == 'twosComplement':
            if binData[0] == '0':
                value = int(binData, 2)
            elif binData[0] == '1':
                value = int(binData, 2) - (1<<len(binData))
        elif paraDefn['encoding'] == 'unsigned':
            value = int(binData, 2)
        if paraDefn['type'] == 'ENUMERATED':
            value = paraDefn['enumSet'][str(value)] 
        return value

    def AddReplies(self, replyFile):
        '''
        Read reply definition. Data structure:
        subsystem (dict) -> service (dict) -> reply (list) -> a parameter of the reply
        Although a command only has one format of reply, it can have different replies. For example, reply of GetRXFrame is [radioservice=25, request=2, RadioReply=0/1/2, RXFrame]. If GetRXFrame fails, the reply only contains the first 3 parameters 
        '''
        self.replies = {'PQ':{}, 'EPS':{}, 'OBC':{}, 'ADCS':{}, 'ADB':{}, 'COMMS':{}, 'LOBE':{}, 'PROP':{}}
        with open(replyFile, encoding = 'utf-8', errors = 'replace') as csvFile:
            csvReader = csv.reader(csvFile)
            firstLine = True
            for row in csvReader:
                if firstLine == True:
                    firstLine = False
                    continue
                if row[3] == 'Reply':
                    subsystem = row[0].replace('Delfi-PQ','').replace(r'/', '')
                    if subsystem == '':
                        subsystem = 'PQ'
                    service = row[1]
                    if service not in self.replies[subsystem].keys():
                        self.replies[subsystem][service] = {}
                    command = row[2]
                    self.replies[subsystem][service][command] = []
                elif row[3] == 'Argument':
                    parameter = row[4]
                    self.replies[subsystem][service][command].append(parameter)
            for subsystem in self.replies:
                for key in self.replies['PQ']:
                    self.replies[subsystem][key] = self.replies['PQ'][key]

    def AddParameters(self, paraFile):
        '''
        Read parameter definitions. The data structure is:
        subsystem (dict) -> parameter (dict) -> attributes of a parameter 
        
        One special parameter is cmdParas, which is a list of parameters used in commands 
        '''
        self.parameters = {'PQ':{}, 'EPS':{}, 'OBC':{}, 'ADCS':{}, 'ADB':{}, 'COMMS':{}, 'LOBE':{}, 'PROP':{}}
        with open(paraFile, encoding = 'utf-8', errors = 'replace') as csvFile:
            csvReader = csv.reader(csvFile)
            firstLine = True
            index = 0
            for row in csvReader:
                if firstLine == True:
                    firstLine = False
                    continue
                subsystem = row[0].replace('Delfi-PQ','').replace(r'/', '')
                if subsystem == '':
                    subsystem = 'PQ'
                if 'cmdParas' not in self.parameters[subsystem]:
                    self.parameters[subsystem]['cmdParas'] = []
                name = row[1]
                self.parameters[subsystem][name] = {}
                # for embedding to neural network
                self.parameters[subsystem][name]['index'] = index 
                index = index + 1
                self.parameters[subsystem][name]['type'] = row[2]
                self.parameters[subsystem][name]['unit'] = row[3]
                if row[4] == 'unknown':
                    self.parameters[subsystem][name]['size'] = 'unknown'
                    continue
                self.parameters[subsystem][name]['size'] = int(row[4])
                self.parameters[subsystem][name]['encoding'] = row[5]
                self.parameters[subsystem][name]['minValue'] = float('NaN')
                if row[6] != '':
                    self.parameters[subsystem][name]['minValue'] = float(row[6])
                self.parameters[subsystem][name]['maxValue'] = float('NaN')
                if row[7] != '':
                    self.parameters[subsystem][name]['maxValue'] = float(row[7])
                self.parameters[subsystem][name]['selectFrom'] = 'Reply Only'
                if row[8] != 'Reply Only':
                    self.parameters[subsystem][name]['selectFrom'] = json.loads(row[8]) 
                    self.parameters[subsystem]['cmdParas'].append(name)
                self.parameters[subsystem][name]['enumSet'] = {}
                if self.parameters[subsystem][name]['type'] == 'ENUMERATED':
                    self.parameters[subsystem][name]['enumSet'] = json.loads(row[9])
            for subsystem in self.parameters:
                for key in self.parameters['PQ']:
                    self.parameters[subsystem][key] = self.parameters['PQ'][key]
            self.parameterNum = index
