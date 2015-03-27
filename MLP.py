import numpy as np
import theano
import theano.tensor as T

from Layer import Layer

class MLP(object):
  
  def __init__(self ,
  	           n_in ,
  	           n_out , 
  	           hidStruct ,
  	           input = None ):

    self.n_in  = n_in   # number of input dimensions 
    self.n_out = n_out  # number of output labels
    self.hidStruct = hidStruct # a list describe the nodes of each hidden layer

    self.struct =  [self.n_in] + self.hidStruct + [self.n_out] # nodes of all layers
    
    #####################################
    # construct multi-layer NNet
    #####################################
    self.layers = []
    for i in range( len(self.struct) - 2 ):
      self.layers.append(
      	                 Layer( 
      	                 	    name = ['layer ' , str(i+1)] ,
      	                        n_in  = self.struct[i]       ,
      	                        n_out = self.struct[i+1]     ,
      	                      )
                        )
    self.layers.append(
                       Layer(
                             name = 'output layer' ,
                             n_in = self.struct[-2] ,
                             n_out = self.struct[-1] ,
                             aFnt = T.nnet.softmax
                       	    )

                      )

    self.params = []
    for layer in self.layers:
      self.params += layer.params
    # print(self.params)


  def forwardProp(self , x):
    for layer in self.layers:
      x = layer.feed(x)
    return x

  def predict(self , x):
  	return self.forwardProp(x)

  def squareError(self , x , y):
    return T.mean((self.predict(x) - y)**2)
  
  def crossEntropyError(self , x , y):
    temp = self.predict(x)
    if self.n_out == 1 :
      return T.mean(T.nnet.binary_crossentropy(temp,y))
    else:
      return T.mean(T.nnet.categorical_crossentropy(temp,y))
    # p_y_given_x = T.nnet.softmax(temp)
    # return -T.mean(T.log(p_y_given_x)[T.arange(y.shape[0]), y])
      
  def getNumberOfHidden(self):
  	print('input:' , str(self.struct[0]))
  	print(self.struct[1:-2])
  	print('output:' , str(self.struct[-1]))
  	return self.struct

  def dAPreTraining(self, x):
    for layer in self.layers:
      layer.doPreTraining(x, 0.4, 0.3)
      x = layer.feed(x)
      tmp = theano.function([],x);
      x = tmp();


