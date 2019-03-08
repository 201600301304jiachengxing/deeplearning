import numpy as np
import torch as tc

#Takes two 1-dimensional arrays and sums the products of all the pairs.
def vectorize_sumproducts(a, b):
    return a.dot(b)

#Takes one 2-dimensional array and apply the relu function on all the values of the array.
def vectorize_Relu(a):
    index = np.where(a>0)
    b = np.zeros(a.shape)
    b[index] = a[index]
    return b

#Takes one 2-dimensional array and apply the derivative of relu function on all the values of the array.
def vectorize_PrimeRelu(a):
    index = np.where(a>0)
    b = np.zeros(a.shape)
    b[index] = 1
    return b

def Slice_fixed_point(a, l, s):
    # Takes one 3-dimensional array with the starting position and the length of the output instances.
    # Your task is to slice the instances from the same starting position for the given length.
    return np.array([a[i][s:s+l] for i in range(len(a))])

def slice_last_point(a, e):
    #Takes one 3-dimensional array with the length of the output instances.
    #Your task is to keeping only the l last points for each instances in the dataset.
    return np.array([a[i][len(a[i])-e:len(a[i])] for i in range(len(a))])

def slice_random_point(a, r):
    b = [len(a[i]) for i in range(len(a))]
    k = np.random.randint(min(b)-r)
    #Takes one 3-dimensional  array  with  the  length  of the output instances.
    #Your task is to slice the instances from a random point in each of the utterances with the given length.
    #Please use function numpy.random.randint for generating the starting position.
    return np.array([a[i][k:k+r] for i in range(len(a))])

def pad_pattern_end(a):
    maxl = max([len(a[i]) for i in range(len(a))])
    #Takes one 3-dimensional array.
    #Your task is to pad the instances from the end position as shown in the example below.
    #That is, you need to pad the reflection of the utterance mirrored along the edge of the array.
    return np.array([np.pad(a[i],((0,maxl-len(a[i])),(0,0)),'symmetric') for i in range(len(a))])

def pad_constant_central(a, val):
    maxl = max([len(a[i]) for i in range(len(a))])
    #Takes one 3-dimensional array with the constant value of padding.
    #Your task is to pad the instances with the given constant value while maintaining the array at the center of the padding.
    return np.array([np.pad(a[i],(((maxl-len(a[i]))//2,maxl-len(a[i])-(maxl-len(a[i]))//2),(0,0)),'constant',constant_values=val) for i in range(len(a))])

#Takes a numpy ndarray and converts it to a PyTorch tensor.
#Function torch.tensor is one of the simple ways to implement it but please do not use it this time.
def numpy2tensor(a):
    return tc.from_numpy(a)

#Takes a PyTorch tensor and converts it to a numpy ndarray.
def tensor2numpy(a):
    return np.array(a)

#you are to implement the function tensor sumproducts that takes two tensors as input.
#returns the sum of the element-wise products of the two tensors.
def Tensor_Sumproducts(a,b):
    return a.dot(b)

#Takes one 2-dimensional tensor and apply the relu function on all the values of the tensor.
def Tensor_Relu(a):
    index = np.where(a>0)
    b = np.zeros(a.shape)
    b[index] = a[index]
    return tc.from_numpy(b).view(a.shape)

#Takes one 2-dimensional tensor and apply the derivative of relu function on all the values of the tensor.
def Tensor_Relu_prime(a):
    index = np.where(a > 0)
    b = np.zeros(a.shape)
    b[index] = a[index]
    return tc.from_numpy(b).view(a.shape)


# test
a = np.arange(16)-8
print(a)

print(vectorize_sumproducts(a,a))
a = a.reshape((4,4))
print(vectorize_Relu(a))
print(vectorize_PrimeRelu(a))

z = np.arange(200).reshape((-1,4))
x = np.array([[z[0],z[1],z[2],z[3]],[z[4],z[5],z[6]],[z[7],z[8],z[9],z[10],z[11],z[12],z[13],z[14]]])
print(Slice_fixed_point(x,2,0))
print(slice_last_point(x,2))
print(slice_random_point(x,2))
print(pad_pattern_end(x))
print(pad_constant_central(x,0))

a = a.reshape((16))
b = numpy2tensor(a)
c = tensor2numpy(b)
print(b)
print(c)
print(Tensor_Sumproducts(b,b))
b = b.view((4,4))
print(Tensor_Relu(b))
print(Tensor_Relu_prime(b))





