import numpy as np
import theano
import theano.tensor as T

from Layer import Layer
from MLP import MLP
import MLPtrainer

hiddenStruct = [5,10]

x = T.matrix("x")

"""
# test layer object
l = Layer(5,3)
f = theano.function([x],l.feed(x))
print(f(np.vstack([[1,-1,2,-2,0],[1,-1,2,-2,0]])))
"""

model = MLP(
	        n_in=3 ,
	        n_out=2 ,
	        hidStruct = hiddenStruct
	       )


input = np.array([[1,2,3],[4,5,6]],dtype=theano.config.floatX)

""" test MLP
f = theano.function([x],model.predict(x))
print(f(input))
"""

label = np.array([[0,1],[1,0]],dtype=theano.config.floatX)



t = MLPtrainer.MLPtrainer(
	       x = input ,
	       y = label ,
	       net = model )
