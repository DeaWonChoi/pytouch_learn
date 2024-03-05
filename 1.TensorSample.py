# https://tutorials.pytorch.kr/beginner/basics/tensorqs_tutorial.html
#
# 텐서(tensor)는 배열(array)이나 행렬(matrix)과 매우 유사한 특수한 자료구조입니다.
# PyTorch에서는 텐서를 사용하여 모델의 입력(input)과 출력(output), 그리고 모델의 매개변수들을 부호화(encode)합니다.
#
# 텐서는 GPU나 다른 하드웨어 가속기에서 실행할 수 있다는 점만 제외하면 NumPy 의 ndarray와 유사합니다.
# 실제로 텐서와 NumPy 배열(array)은 종종 동일한 내부(underly) 메모리를 공유할 수 있어 데이터를 복사할 필요가 없습니다.
# (NumPy 변환(Bridge) 참고) 텐서는 또한 (Autograd 장에서 살펴볼) 자동 미분(automatic differentiation)에 최적화되어 있습니다.
# ndarray에 익숙하다면 Tensor API를 바로 사용할 수 있을 것입니다. 아니라면, 아래 내용을 함께 보시죠!

import torch
import numpy as np

data = [[1, 2], [3, 4]]

# 데이터로부터 직접(directly) 생성하기
# 데이터로부터 직접 텐서를 생성할 수 있습니다. 데이터의 자료형(data type)은 자동으로 유추합니다.
x_data = torch.tensor(data)
print(f"x_data: \n {x_data} \n")

# NumPy 배열로부터 생성하기
# 텐서는 NumPy 배열로 생성할 수 있습니다. (그 반대도 가능합니다 - NumPy 변환(Bridge) 참고)
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
print(f"x_np: \n {x_np} \n")

# 다른 텐서로부터 생성하기:
# 명시적으로 재정의(override)하지 않는다면, 인자로 주어진 텐서의 속성(모양(shape), 자료형(datatype))을 유지합니다.
x_ones = torch.ones_like(x_data)  # x_data의 속성을 유지합니다.
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float)  # x_data의 속성을 덮어씁니다.
print(f"Random Tensor: \n {x_rand} \n")

# 무작위(random) 또는 상수(constant) 값을 사용하기:
# shape 은 텐서의 차원(dimension)을 나타내는 튜플(tuple)로, 아래 함수들에서는 출력 텐서의 차원을 결정합니다.
shape = (2, 3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor} \n")

# GPU가 존재하면 텐서를 이동합니다
if torch.cuda.is_available():
    tensor = torch.ones(4, 4)
    tensor = tensor.to("cuda")

# NumPy식의 표준 인덱싱과 슬라이싱:
tensor = torch.ones(4, 4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:, 1] = 0
print(tensor)

# 텐서 합치기 torch.cat 을 사용하여 주어진 차원에 따라 일련의 텐서를 연결할 수 있습니다. torch.cat 과 미묘하게 다른 또 다른 텐서 결합 연산인 torch.stack 도 참고해보세요.
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

# 두 텐서 간의 행렬 곱(matrix multiplication)을 계산합니다. y1, y2, y3은 모두 같은 값을 갖습니다.
# ``tensor.T`` 는 텐서의 전치(transpose)를 반환합니다.
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)

# 요소별 곱(element-wise product)을 계산합니다. z1, z2, z3는 모두 같은 값을 갖습니다.
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)

# 단일-요소(single-element) 텐서 텐서의 모든 값을 하나로 집계(aggregate)하여 요소가 하나인 텐서의 경우, item() 을 사용하여 Python 숫자 값으로 변환할 수 있습니다:
agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))

