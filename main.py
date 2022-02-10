import torch
import numpy as np

data = [[1,2], [3,4]]
np_data = np.array(data)
x_data = torch.tensor(data) # 기본 list를 tensor로 변환 시킬 수 있고,
x_npdata = torch.from_numpy(np_data) # 필요에 따라서는 numpy array를 변환시킬 수 있다.

print(type(x_data))
print(type(x_npdata))

shapes = (2,3,)

rand_tensor = torch.rand(shapes) # torch로 tensor를 생성할 떄는 임의의 tuple을 사용해 만들수 있다.
ones_tensor = torch.ones(shapes)
zeros_tensor = torch.zeros(shapes)

