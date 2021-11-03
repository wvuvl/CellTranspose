import numpy as np
import torch
import torch.nn.functional as F


X = torch.tensor([  [1,2,3,4],
                    [1,2,3,4],
                    [1,2,3,4],
                    [1,2,3,4]])
print(X)
X = F.pad(X,(0,0,2,2))
print(X)

print(X.shape)