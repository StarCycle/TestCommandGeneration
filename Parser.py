import csv
import json

def AddParameters(paraFile):
    '''
    Return a dict of parameters. Each parameter has its own dict.
    '''
    parameters = {}
    with open(paraFile, encoding = 'utf-8', errors = 'replace') as csvFile:
        csvReader = csv.reader(csvFile)
        firstLine = True
        for row in csvReader:
            if firstLine == True:
                firstLine = False
                continue
            name = row[1]
            parameters[name] = {}
            parameters[name]['subsystem'] = row[0]
            parameters[name]['type'] = row[2]
            parameters[name]['size'] = int(row[3])
            parameters[name]['selectFrom'] = json.loads(row[4]) 
            parameters[name]['enumSet'] = {}
            if row[5] != '':
                parameters[name]['enumSet'] = json.loads(row[5])
    return parameters

parameters = AddParameters('para.csv')
print('here')
