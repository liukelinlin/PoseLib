import numpy as np
import poselib


x1 = np.random.rand(3,3)
x2 = np.random.rand(3,3)

res = poselib.essential_matrix_5pt(x1, x2)
print(res)


