import numpy as np
import theano
import theano.tensor as T

import os
import sys

import cPickle


class Layer(object):
  """
  Layer Object:

  parameters:

  do the weight initialization itself
  initialization method reference:
  http://neuralnetworksanddeeplearning.com/chap3.html#weight_initialization


  """
  def __init__(self ,  
               name = 'some_layer', 
               n_in , 
               n_out , 
               aFnt = T.nnet.sigmoid ):
    
    self.name = name  
    self.n_in = n_in
    self.n_out = n_out
    
    ###################################
    # weight and bias initialization #
    ###################################
    rng = np.random.RandomState(1234)
    W_values = numpy.asarray(
        rng.uniform(
            low=-numpy.sqrt(6. / (n_in + n_out)),
            high=numpy.sqrt(6. / (n_in + n_out)),
            size=(n_in, n_out)
        ),
        dtype=theano.config.floatX
    )
    if activation == theano.tensor.nnet.sigmoid:
      W_values *= 4
    self.W = theano.shared(value=W_values, name='W', borrow=True)

    b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
    self.b = theano.shared(value=b_values, name='b', borrow=True)

    self.params = [self.W , self.b]
    
    ############################################
    # calculate output value if input is given #
    ############################################
    self.aFnt = aFnt
    self.input = None
    self.output = None


  def feed(self , input):
    if len(input) != self.n_in :
      raise TypeError(self.name , ": wrong input dimension")
    
    self.input = input
    
    a = self.input
    W = self.W
    b = self.b
    
    output = T.dot(a , W) + b
    
    if self.aFnt is None:
      self.output = output
    else:
      self.output = self.aFnt(output)

    return self.output
  
  def crossEntropyError(self , y):
    '''
    negative log likelihood
    use softmax function
    '''
    if not self.output:
      raise TypeError(self.name , ': feed something first')
    
    temp = T.dot(input, self.W) + self.b
    p_y_given_x = T.nnet.softmax(temp)
    return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])


  def squareError(self , y):
    if not self.output:
      raise TypeError(self.name , ': feed something first')
    return T.mean((self.output - y)**2)
    


  def setActivationFunction(self , aFnt):
    self.aFnt = aFnt 

  def setWeight(self , W):
    """
    change the weight matrix
    
    use it if you want to initialize
    the weight matrix outside the object
    """
    self.W = W

  def setBias(self , b):
    self.b = b
  
  def setName(self , name):
    self.name = name