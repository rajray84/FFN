'''
Objective functions can be implemented in this file

Author:
    Yi-Qi Hu

Time:
    2016.6.13
'''

'''
 This program is free software; you can redistribute it and/or
 modify it under the terms of the GNU General Public License
 as published by the Free Software Foundation; either version 2
 of the License, or (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program; if not, write to the Free Software
 Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.

 Copyright (C) 2015 Nanjing University, Nanjing, China
'''

import math
import time
import onnx
import onnxruntime as rt
from util import predictWithOnnxruntime, propCheck
import numpy as np


# FFN evaluation for continue optimization
def nneval(onnxModel,inVals,inpDtype,inpShape,inpSpecs,target,objType):
   flattenOrder='C'
   inputs = np.array(inVals, dtype=inpDtype)
   inputs = inputs.reshape(inpShape, order=flattenOrder) # check if reshape order is correct
   assert inputs.shape == inpShape

   output = predictWithOnnxruntime(onnxModel, inputs)
   flatOut = output.flatten(flattenOrder) # check order, 'C' for row major order

   # objType = 0 -> maximization
   # objType = 1 -> minimization

   retVal=propCheck(inVals,inpSpecs,flatOut)
   if (objType == 0):
      flatOut[target] = -flatOut[target]

   return retVal, flatOut[target]

