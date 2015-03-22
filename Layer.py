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
               n_in , 
               n_out , 
               name = 'some_layer',
               aFnt = T.nnet.sigmoid ):
    
    self.name = name  
    self.n_in = n_in
    self.n_out = n_out
    
    ###################################
    # weight and bias initialization #
    ###################################
    rng = np.random.RandomState()
    W_values = np.asarray(
        rng.uniform(
            low=-np.sqrt(6. / (n_in + n_out)),
            high=np.sqrt(6. / (n_in + n_out)),
            size=(n_in, n_out)
        ),
        dtype=theano.config.floatX
    )
    if aFnt == T.nnet.sigmoid:
      W_values *= 4
    self.W = theano.shared(value=W_values, name='W', borrow=True)

    b_values = np.zeros((n_out,), dtype=theano.config.floatX)
    self.b = theano.shared(value=b_values, name='b', borrow=True)

    self.params = [self.W , self.b]
    
    self.aFnt = aFnt
    self.input = None
    self.output = None


  def feed(self , input):
    """
    if len(input) != self.n_in :
      raise TypeError(self.name , ": wrong input dimension")
    """
    
    W = self.W
    b = self.b
    
    output = T.dot(input , W) + b
    
    if self.aFnt is None:
      self.output = output
    else:
      self.output = self.aFnt(output)

    return self.output


  
  def setActivationFunction(self , aFnt):
    self.aFnt = aFnt 

  def setWeight(self , W):
    """
    change the weight matrix

    use it if you want to initialize
    the weight matrix outside the object
    """
    self.W = theano.shared(value = W.astype(theano.config.floatX),
                           name = 'W' ,
                           borrow = True)
    self.params = [self.W, self.b]

  def setBias(self , b):
    self.b = theano.shared(value = b.astype(theano.config.floatX) ,
                           name = 'b' ,
                           borrow = True)
    self.params = [self.W, self.b]
  
  def setName(self , name):
    self.name = name

  """
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
  """