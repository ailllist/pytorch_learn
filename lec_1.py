import torch
import numpy as np
import time

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

# tensor는 다음과 같은 속성을 가진다. tensor.shape, tensor.dtype, tensor.device
tensor = torch.rand(3,4)
tensor.to("cuda")

print(f"Shape of tensor : {tensor.shape}") 
print(f"dtype of tensor : {tensor.dtype}")
print(f"Device tensor is stored on : {tensor.device}")

tensor = torch.ones(4,4)

if torch.cuda.is_available():
    tensor = tensor.to("cuda")
    print("GPU")

print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}") # ... ~= : 랑 비슷함.
tensor[:, 1] = 0

# tensor 연산

# 결합 연산
t1 = torch.cat([tensor, tensor, tensor], dim=1) # tensor간 결합, [1,2,3,...] : dim=0 / [[1,2,3], [4,5,6], ...] : dim=1
tensor_T = tensor.T # 전치행렬

# 행렬 곱
y1 = tensor @ tensor.T # 행렬곱 표현식1
y2 = tensor.matmul(tensor.T) # 행렬곱 표현식2 tensor @ tensor.T
y3 = torch.ones_like(tensor) # tensor와 같은 크기를 가진 배열 y3를 만듬.
torch.matmul(tensor, tensor.T, out=y3) # 행렬곱 표현식3
y4 = torch.matmul(tensor, tensor.T) # 행렬곱 표현식4

# tensor.matmul와 torch.matmul둘다 return값이 있지만, torch.matmul은 output tensor를 지정할 수 있다.

# 요소 곱
z1 = tensor * tensor # 요소곱 표현식1
z2 = tensor.mul(tensor) # 요소곱 표현식2
z3 = torch.ones_like(tensor) # 밑에 줄을 위한 코드
torch.mul(tensor, tensor, out=z3) # 요소곱 표현식2
# tensor.matmul과 유사한듯.

agg = tensor.sum() # tensor의 모든 요소 합 -> 1 by 1행렬을 만든다.

# 1 by 1행렬 특이 연산
agg_item = agg.item() # 1 by 1 tensor의 경우 item함수를 사용해 내부 요소에 접근 가능하다.
print(agg_item)

# 바꿔치기 연산 (피연산자가 삭제됨. 자기 자신을 리턴하는 연산)
print(tensor)
tensor.add_(5) # 바꿔치기 연산의 특징: 연산자 뒤어 _를 붙이면 된다.
print(tensor)
# 주의점. 바꿔치기 연산을 진행하면 tensor변수 자체를 바꾸는 것이기 때문에 메모리 절약 측면에서는 효율적이지만, 히스토리가 삭제되어 도함수를 계산할때 어렵다고 한다.

# numpy와의 연계

n = np.ones(5)
t = torch.from_numpy(n)

np.add(n, 1, out=n) # t는 얕은 복사이다. 즉 n이 바뀌면 t도 같이 변함.
print(t)
print(n)