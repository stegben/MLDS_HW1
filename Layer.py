import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

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
    self.theano_rng = RandomStreams(rng.randint(2 ** 30));
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
    self.b_prime = theano.shared(value = np.zeros(n_in,
                                        dtype=theano.config.floatX), name='bvis')
    self.params = [self.W , self.b]
    
    self.aFnt = aFnt
    self.input = None
    self.output = None
    self.z = None


  def feed(self , input):
    """
    if len(input) != self.n_in :
      raise TypeError(self.name , ": wrong input dimension")
    """
    self.input = input
    W = self.W
    b = self.b
    
    self.z = T.dot(self.input , W) + b
    
    if self.aFnt is None:
      self.output = self.z
    else:
      self.output = self.aFnt(self.z)

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


# Stacked Autoencoder related function
  def getCorruptedInput (self, x, corruption_level):
    return self.theano_rng.binomial(size=x.shape, n=1, p=1 - corruption_level) * x

  def getHiddenValues (self, x):
    return self.aFnt(T.dot(x, self.W) + self.b)

  def getReconstructedInput(self, hidden ):
       """ Computes the reconstructed input given the values of the hidden layer """
       return  self.aFnt(T.dot(hidden, self.W.T) + self.b_prime)
  
  def getCostUpdates(self, x, corruption_level, learning_rate):
       """ This function computes the cost and the updates for one trainng
       step of the dA """

       tilde_x = self.getCorruptedInput(x, corruption_level)
       y = self.getHiddenValues( tilde_x)
       z = self.getReconstructedInput(y)
       # note : we sum over the size of a datapoint; if we are using minibatches,
       #        L will  be a vector, with one entry per example in minibatch
       #L = -T.sum(x * T.log(z) + (1 - x) * T.log(1 - z), axis=1 )
       dv = x - z
       L = dv.norm(2) ** 2
       # note : L is now a vector, where each element is the cross-entropy cost
       #        of the reconstruction of the corresponding example of the
       #        minibatch. We need to compute the average of all these to get
       #        the cost of the minibatch
       cost = T.mean(L)

       # compute the gradients of the cost of the `dA` with respect
       # to its parameters
       gparams = T.grad(cost, [self.W , self.b , self.b_prime])
       # generate the list of updates
       updates = []
       for param, gparam in zip([self.W, self.b, self.b_prime], gparams):
           updates.append((param, param - learning_rate * gparam))

       return (cost, updates)

  def doPreTraining(self, training_set_x, corruption_level, learning_rate):
    x = T.dmatrix('x');
    cost, updates = self.getCostUpdates(x, corruption_level, learning_rate)
    train_da = theano.function([x], cost, updates=updates)
    for epoch in xrange(12):
        # go through trainng set
        #c = []
        #c.append(train_da(training_set_x))
        train_da(training_set_x)
        #if(epoch == 11):
        #  print 'In Pre-training epoch %d, cost ' % epoch, np.mean(c)



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