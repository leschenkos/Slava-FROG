"""error class originally written for the D-scan retrieval"""

class SL_exception(Exception):
    def __init__(self,text):
        self.Message=text


class ReadError(SL_exception):
    def __init__(self,oserror):
            self.Message=str(oserror)
            
class CalibrationError():
    pass


    
        