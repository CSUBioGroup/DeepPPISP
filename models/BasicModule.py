#-*- encoding:utf-8 -*-

import torch as t
import time

class BasicModule(t.nn.Module):

    def __init__(self):
        super(BasicModule,self).__init__()
        self.model_name = str(type(self))
        
    def load(self,path):

        self.load_state_dict(t.load(path))
        
    def save(self,name=None):

        
        if name is None:
            prefix = ""
            name = time.strftime("%y%m%d_%H:%M:%S.pth".format(prefix))
            
        t.save(self.state_dict(),name)
        return name
