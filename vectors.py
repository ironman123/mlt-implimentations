#%%
import numpy as np
x=np.array([1,2,3])
y=np.array([4,5,6])


#%%
#vector addition
x+y

#%%
#scaling vectors
3*x
#%%
#elementwise multiplication
x*y

# %%
#elementwise function of vectors
#power
x**3
# %%
#log10 (base 10)
z=np.array([10, 100, 1000, 10_000, 100_000])
np.log10(z) #this gives float64 output i.e. array([1., 2., 3.])

# %%
#type is numpy.ndarray
type(z)
# %%
#output is [10 100 1000]
print(z)
# %%
z.dtype
# %%
e = [1,np.e,np.e ** 2,np.e ** 3, np.pi]
np.log(e)
# %%
#addition to each element
c = 1
x+c
# %%
#now lets do 5x-2 where x is a vector
5 * x - 2
# %%
#dot product
np.dot(x,y)
# %%
#alternate way of dot product
x @ y
# %%
#vector of zeros of length n = 5
np.zeros(5)
# %%
#vector of ones of length n = 3
np.ones(3)
# %%
#now a vector of cs of length n = 10
5 * np.ones(10)
# %%
#for sequence of numbers (1,6)-> [1,2,3,4,5]
np.arange(2,6)
# %%
#NORMS:
#L1
np.linalg.norm(x , ord=2) #ridge
# %%
#L2
np.linalg.norm(x,ord=1) #lasso
# %%
#shape & dimension
print(x.shape)
print(x.ndim)
# %%
a = np.array([[1,2],[3,4],[5,6]])
print(a.shape)
print(a.ndim)
# %%
b = np.array([[[1,0,0],[0,1,0],[0,0,1]],[[1,2,3],[4,5,6],[7,8,9]]])
print(b.shape)
print(b.ndim)
#ndim tells what's the dimension of the matrix formed, i.e it is a 1D vector or 2D matrix or 3D matrix or a 4D tesseract or a 0D scaler

# %%
