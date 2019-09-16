from numba import cuda
from numba import vectorize, float64
import math

@cuda.jit(device=True, inline=True)
def subtract(x, y, store):
    for i in range(len(x)):
        store[i] = x[i] - y[i]

@cuda.jit(device=True, inline=True)
def multiply(x, y, store):
    for i in range(len(y)):
        store[i] = x * y[i]

@cuda.jit(device=True, inline=True)
def multiply_element_wise(x, y, store):
    for i in range(len(y)):
        store[i] = x[i] * y[i]

@cuda.jit(device=True, inline=True)
def multiply_element_wise_shared(x, y, store):
    for i in range(len(x)):
        store[i] = x[i] * y[cuda.threadIdx.x + cuda.blockDim.x * i]

@cuda.jit(device=True, inline=True)
def add(x, y, store):
    for i in range(len(x)):
        store[i] = x[i] + y[i]

@cuda.jit(device=True, inline=True)
def zeros(x, store):
    for i in range(len(x)):
        store[i] = 0

@cuda.jit(device=True, inline=True)
def equalize(x, y):
    for i in range(len(y)):
        x[i] = y[i]

@cuda.jit(device=True, inline=True)
def qtilde_to_primitive_cuda(qtilde, gamma, shared):

    q1 = qtilde[0]
    q2 = qtilde[1]
    q3 = qtilde[2]
    q4 = qtilde[3]

    beta = -q4*0.5

    temp = 0.5/beta

    u1 = q2*temp
    u2 = q3*temp

    temp1 = q1 + beta*(u1*u1 + u2*u2)
    temp2 = temp1 - (math.log(beta)/(gamma-1))
    rho = math.exp(temp2)
    pr = rho*temp

    shared[cuda.threadIdx.x + cuda.blockDim.x * 4] = u1
    shared[cuda.threadIdx.x + cuda.blockDim.x * 5] = u2
    shared[cuda.threadIdx.x + cuda.blockDim.x * 6] = rho
    shared[cuda.threadIdx.x + cuda.blockDim.x * 7] = pr

@cuda.reduce
def sum_reduce(a, b):
    return a + b