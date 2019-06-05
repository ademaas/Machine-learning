#simple machine learnijng with single hidden layer
import numpy as np 

def sigmoid (x):
    return 1/(1+np.exp(-x))

class MBO:
    def __init__(self,weight,bias):
        self.weight = weight
        self.bias = bias

    def forward_propagation(self,input):
        total= input*self.weight + self.bias
        return sigmoid(total)

weight = np.array([1,1])
bias = 4

mbc = MBO(weight,bias)
inputs = np.array([2,3])
forward = mbc.forward_propagation(inputs)

print(forward)
    

