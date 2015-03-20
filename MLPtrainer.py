import numpy as np
import theano
import theano.tensor as T

import MLP

"""


"""
def MLPtrainer(learning_rate = 0.01,
  	           momentum = 0.9,
  	           L1 = 0.0,
  	           L2 = 0.0,
  	           batch_size = 1,
  	           epoch = 1,
  	           x ,
  	           y ,
  	           net ):
  regL1 = []
  for layer in net.layers:
  	regL1 += abs(layer.W).mean()

  regL2 = []
  for layer in net.layers:
  	regL2 += (layer.W ** 2).mean()

  cost = (
          net.crossEntropyError(x,y)
        + L1 * regL1
        + L2 * regL2
    )

  gparams = [T.grad(cost, param) for param in net.params]

  updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(net.params, gparams)
  ]

  train_model = theano.function(
             inputs = [x,y],
             outputs = cost ,
             updates = updates

  	)


  print('start training...')
  itr=0
  
  while itr < epoch :
    print('epoch : ',str(itr))
    net.




    epoch -= 1
    itr +=1

def 


def gradient_updates_momentum(cost, params, learning_rate, momentum):
    '''
    Compute updates for gradient descent with momentum
    
    :parameters:
        - cost : theano.tensor.var.TensorVariable
            Theano cost function to minimize
        - params : list of theano.tensor.var.TensorVariable
            Parameters to compute gradient against
        - learning_rate : float
            Gradient descent learning rate
        - momentum : float
            Momentum parameter, should be at least 0 (standard gradient descent) and less than 1
   
    :returns:
        updates : list
            List of updates, one for each parameter
    '''
    # Make sure momentum is a sane value
    assert momentum < 1 and momentum >= 0
    # List of update steps for each parameter
    updates = []
    # Just gradient descent on cost
    for param in params:
        # For each parameter, we'll create a param_update shared variable.
        # This variable will keep track of the parameter's update step across iterations.
        # We initialize it to 0
        param_update = theano.shared(param.get_value()*0., broadcastable=param.broadcastable)
        # Each parameter is updated by taking a step in the direction of the gradient.
        # However, we also "mix in" the previous step according to the given momentum value.
        # Note that when updating param_update, we are using its old value and also the new gradient step.
        updates.append((param, param - learning_rate*param_update))
        # Note that we don't need to derive backpropagation to compute updates - just use T.grad!
        updates.append((param_update, momentum*param_update + (1. - momentum)*T.grad(cost, param)))
    return updates