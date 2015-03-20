import numpy as np
import theano
import theano.tensor as T

import cPickle


class Layer(object):

  def __init__(self ,  
               input ,  
               n_in , 
               n_out , 
               W  = None , 
               b = None ,
               aFnt = T.nnet.sigmoid ):
    '''
    for input layer, assign aFnt = None
    '''
    self.input = input

    
    # weight initialization
    rng = np.random.RandomState(1234)
    if W is None:
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

        W = theano.shared(value=W_values, name='W', borrow=True)

    if b is None:
        b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
        b = theano.shared(value=b_values, name='b', borrow=True)

    self.W = W
    self.b = b

    output = T.dot(input , W) + b

    if aFnt is None:
      self.output = output
    else:
      self.output = aFnt(output)
  def getOutput(self):
    return self.output
  def setInput(self,input):
    self.input = input
