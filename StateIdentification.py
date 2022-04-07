
class StateIdentification:

    def NewState(self, oldState, command, response):
        newState = oldState.copy()
        method = getattr(self,'command_' + command['Command'], None)
        if callable(method):
            return method(newState, command, response)
        else:
            return newState

    def command_Ping(self, newState, command, response):
        newState['Ping'] = response['Request']
        return newState

    def command_Housekeeping(self, newState, command, response):
        newState['Housekeeping'] = response['Request']
        return newState

    def command_SimpleRadioOps(self, newState, command, response):
        newState[command['SimpleRadioOps']] = response['RadioReply']
        return newState

    def command_GetRXFrameNr(self, newState, command, response):
        newState['GetRXFrameNr'] = response['RadioReply']
        if response['RXFrameNumber'] == 0:
            newState['RXFrameNumber'] = 'No RX Frame'
        elif response['RXFrameNumber'] < 80:
            newState['RXFrameNumber'] = 'Not full'
        elif response['RXFrameNumber'] == 80:
            newState['RXFrameNumber'] = 'Full'
        return newState

    def command_GetRXFrame(self, newState, command, response):
        newState['GetRXFrame'] = response['RadioReply']
        return newState

    def command_GetRX_RSSI(self, newState, command, response):
        newState['GetRXRSSI'] = response['RadioReply']
        return newState

    def command_GetRXLast_RSSI(self, newState, command, response):
        newState['GetRXLastRSSI'] = response['RadioReply']
        return newState

    def command_GetRX_Freq_Error(self, newState, command, response):
        newState['GetRXFreqError'] = response['RadioReply']
        return newState

    def command_SendFrame(self, newState, command, response):
        newState['SendFrame'] = response['RadioReply']
        return newState

    def command_SetTXIdleState(self, newState, command, response):
        newState['SetTXIdleState'] = response['RadioReply']
        if newState['SetTXIdleState'] == 'No error':
            newState['Idle'] = command['Idle']
        return newState

    def command_SetTXBitRate(self, newState, command, response):
        newState['SetTXBitRate'] = response['RadioReply']
        if newState['SetTXBitRate'] == 'No error':
            newState['TXBitRate'] = command['TXBitRate']
        return newState

    def command_SetRXBitRate(self, newState, command, response):
        newState['SetRXBitRate'] = response['RadioReply']
        if newState['SetRXBitRate'] == 'No error':
            newState['RXBitRate'] = command['RXBitRate']
        return newState

    def command_SetRadioPA(self, newState, command, response):
        newState['SetRadioPA'] = response['RadioReply']
        if newState['SetRadioPA'] == 'No error':
            newState['PAMode'] = command['PAMode']
        return newState

    def command_SetTXFreq(self, newState, command, response):
        newState['SetTXFreq'] = response['RadioReply']
        if newState['SetTXFreq'] == 'No error':
            newState['TXFreq'] = command['TXFreq']
        return newState

    def command_SetRXFreq(self, newState, command, response):
        newState['SetRXFreq'] = response['RadioReply']
        if newState['SetRXFreq'] == 'No error':
            newState['RXFreq'] = command['RXFreq']
        return newState

    def command_SetTXPower(self, newState, command, response):
        newState['TXPower'] = command['Power']
        return newState
