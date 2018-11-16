
import numpy as np
from random import shuffle

input_layer = np.zeros((1, 3), dtype=np.float)
output_layer = np.zeros((4, 1), dtype=np.float)


def input_data(a, b, c):
    a_f = np.array((a + 100) / 200, dtype=np.float)
    b_f = np.array((b + 100) / 200, dtype=np.float)
    c_f = np.array((c + 100) / 200, dtype=np.float)
    return np.array([[a_f, b_f, c_f]])

def solver(a, b, c):
    x1 = (np.sqrt(b*b - 4*a*c + 0j) - b) / (2*a)
    x2 = (-np.sqrt(b*b - 4*a*c + 0j) - b) / (2*a)

    return np.array([x1.real, x2.real, x1.imag, x2.imag])


def frange(x, y, jump):
    while x < y:
        yield x
        x += jump
        
def sigmoid(x, deriv=False):
    if deriv == True:
        return sigmoid(x)*(1-sigmoid(x))
    
    return np.array(1/(1 + np.exp(-x)), dtype=np.float32)

def inv_sigmoid(x):
    return -np.log(1 / x - 1)

a = [i for i in frange(-100, 100, 0.1)]
b = [i for i in frange(-100, 100, 0.1)]
c = [i for i in frange(-100, 100, 0.1)]

a_r = a; b_r = b; c_r = c
shuffle(a_r) 
shuffle(b_r)
shuffle(c_r)

x_data = []

for j in range(10):
    x_data.extend([[a_r[i], b_r[i], c_r[i]] for i in range(2000)])
    shuffle(a_r)
    shuffle(b_r)
    shuffle(c_r)

y_data = [solver(*i) for i in x_data]

# print(x_data[1], y_data[1], solver(x_data[1][0], x_data[1][1], x_data[1][2]))

class network:
    def __init__(self):
        self.syn0 = 2*np.random.random((3, 4)) - 1
        self.l1 = 2*np.random.random((1, 4)) - 1
        self.l1_delta = 1
        self.l1_error = np.array([1])
    
    def run(self, a, b, c):
        self.l0 = input_data(a, b, c)
        self.l1 = sigmoid(np.dot(self.l0, self.syn0))
        return self.l1
    
    def backpropagation(self, x_d, y_d):
        for i in range(len(x_d)):
            n = 0
            self.run(x_d[i][0], x_d[i][1], x_d[i][2])
            self.l1_error = sigmoid(y_d[i]) - self.l1
            self.l1_delta = self.l1_error.dot(sigmoid(self.l1, True).transpose())
            while abs(self.l1_error.sum()) > 0.0001:
                n += 1
                
                self.run(x_d[i][0], x_d[i][1], x_d[i][2])
                self.l1_error = sigmoid(y_d[i]) - self.l1
                self.l1_delta = self.l1_error.dot(sigmoid(self.l1, True).transpose())
#                 if i % 1000 == 0:
#                     print('l0', self.l0)
#                     print('l1', self.l1)
#                     print('syn0', self.syn0)
#                     print('l1_err', self.l1_error)
#                 print('l1_delta', self.l1_delta)
#                     print('l1 unsigm', np.dot(self.l0, self.syn0)+np.dot(self.l0.transpose(), self.l1_delta))
#                 print('syn0_delta', np.dot(self.l0.transpose(), self.l1_delta))
                self.syn0 += np.dot(self.l0.transpose(), self.l1_delta)
                if n % 100 == 1:
                    print(self.l1_error)
                    print(self.l1, sigmoid(y_d[i]))
                    print(sigmoid(self.l1, True).transpose())
            if i % 100 == 1:
                print(abs(self.l1_error.sum()))#, inv_sigmoid(self.l1_error.sum()))

net = network()
net.backpropagation(x_data, y_data)

net.run(1, 2, 3)/2
sigmoid(solver(1, 2, 3))

