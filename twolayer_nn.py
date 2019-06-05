import numpy as np 
import sample_nn as s

class Twolayer_nn:
    def __init__(self):
        weight = np.array([0,1])
        bias = 0
        self.h1 =  s.MBO(weight,bias)
        self.h2 = s.MBO(weight,bias)
        self.ol = s.MBO(weight,bias)

    def forward_prop(self,inputs):
        out1=self.h1.forward_propagation(inputs)
        out2= self.h2.forward_propagation(inputs)
        out12 = self.ol.forward_propagation(np.array([out1,out2]))
        return out12

twol = Twolayer_nn()
inputs = np.array([2,3])
print(twol.forward_prop(inputs))
    