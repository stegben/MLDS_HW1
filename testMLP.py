import numpy as np
import theano
import theano.tensor as T


from MLP import MLP
import MLPtrainer

hiddenStruct = [5]
model = MLP(
	        n_in=3 ,
	        n_out=2 ,
	        hidStruct = hiddenStruct
	       )

N=20
np.random.seed(1234)
input = np.array([1,2,3],dtype=theano.config.floatX)
label = np.array([0,1],dtype=theano.config.floatX)

print(model.predict(np.array([2,3,4])))

t = MLPtrainer.MLPtrainer(
	       x = input ,
	       y = label ,
	       net = model )
