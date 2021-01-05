# -*- coding: utf-8 -*-
import datetime

class Logger(object):
    __instance = None 
    __init_flag = False

    def __init__(self, loggerFileName = './log.txt'):
        if not Logger.__init_flag:
            self.loggerFileName = loggerFileName
            self.print3('======== logger created ========')
            Logger.__init_flag = True

    def print3(self, string):
        print(string)
        
        f = open(self.loggerFileName, 'a')
        f.write(str(datetime.datetime.now()) + ' >> ' + string + '\n')
        f.close()

    def __new__(cls, *args, **kwargs):
        if cls.__instance == None:
            cls.__instance = object.__new__(cls)
            return cls.__instance
        else:
            return cls.__instance