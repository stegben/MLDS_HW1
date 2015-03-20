import numpy as np
import theano
import theano.tensor as T

import Layer

class MLP(object):
  
  def __init__(self ,
  	           n_in ,
  	           n_out , 
  	           hidStruct ,
  	           input = None ):

    self.n_in  = n_in   # number of input dimensions 
    self.n_out = n_out  # number of output labels
    self.hidStruct = hidStruct # a list describe the nodes of each hidden layer

    self.struct = [ [self.n_in] ,
                     self.hidden ,
                    [self.n_out] ] # nodes of all layers
    
    #####################################
    # construct multi-layer NNet
    #####################################
    self.layers = []
    for i in renge( len(self.struct) - 1 )
      self.layers.append(
      	                 Layer( 
      	                 	    name = ['layer ' , str(i+1)] ,
      	                        n_in  = self.struct[i]       ,
      	                        n_out = self.struct[i+1]     ,
      	                      )
                        )
    self.layers[-1].setName('output layer')

    self.params = []
    for layer in self.layers:
      self.params += layer.params

    


  def forwardProp(self , x):
    for layer in self.layers:
      x = layer.feed(x)
    return x

  def predict(self , x):
  	return self.forwardProp(x)

  def squareError(self , x , y):
    return T.sum((self.output(x) - y)**2)

  def crossEntropyError(self , x , y):

    
  def getNumberOfHidden(self):