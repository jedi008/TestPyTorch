import numpy as np

arr = np.array([[1, 2 , 4, 5 , 6], [1, 3, 4, 5, 6], [1, 2 , 4, 5 , 6]]) # axis为0时：按照从上往下方向进行唯一化（返回二维数组的唯一行） 相当于二维数组去重
result2 = np.unique(arr,axis=0)
print(type(result2))  # <class 'numpy.ndarray'>
print(result2)  # [1 2 3]