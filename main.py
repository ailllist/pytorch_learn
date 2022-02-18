import torch
import numpy as np

data = [[1,2], [3,4]]
np_data = np.array(data)
x_data = torch.tensor(data) # 기본 list를 tensor로 변환 시킬 수 있고,
x_npdata = torch.from_numpy(np_data) # 필요에 따라서는 numpy array를 변환시킬 수 있다.
xf_data = torch.tensor(data, dtype=torch.float)
# 이때 dtype를 명시해서 자료형을 변경해줄 수 있다.
xfx_data = torch.tensor(xf_data, dtype=torch.int)
# 위와 같이 tensor의 자료형 변환에 사용할 수도 있다. 
# list, ndarray, tensor -> tensor

# print(type(x_data))
# print(type(x_npdata))
# print(type(xf_data))
# print(xfx_data, xf_data)

# 다른 tensor를 통한 tensor생성
x_ones = torch.ones_like(x_data)
# x_rand = torch.rand_like(x_data)

# torch로 tensor를 생성할 떄는 임의의 tuple을 사용해 만들수 있다.
shapes = (2,3,)

rand_tensor = torch.rand(shapes) # 랜덤배열 
ones_tensor = torch.ones(shapes) # 1로 구성된 배열
zeros_tensor = torch.zeros(shapes) # 0 배열
# *args가 포함되어있다.
print(rand_tensor)
print(ones_tensor)
print(zeros_tensor)

# print(torch.cuda.is_available())

if torch.cuda.is_available():
    tensor = rand_tensor.to("cuda")
    print("GPU")

tensor = torch.rand(3,4)
print(f"Shape of tensor : {tensor.shape}")
print(f"dtype of tensor : {tensor.dtype}")
print(f"Device tensor is stored on : {tensor.device}")