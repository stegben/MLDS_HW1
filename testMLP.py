import numpy as np
import theano
import theano.tensor as T

import MLP
import MLPtrainer

hiddenStruct = [128]
model = MLP(
	        n_in=69 ,
	        n_out=48 ,
	        hidStruct = hiddenStruct
	       )

N=20
np.ramdom.seed(1234)
input = 
label = 

MLPtrainer(
	       net = model ,
	       x = X ,
	       y = Y 
	      )
