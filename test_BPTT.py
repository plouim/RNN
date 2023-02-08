import numpy as np

class func:

    def __init__(self, A, B, activation_func, cost_func):
        self.A = A
        self.B = B
        self.cost_func = cost_func
        self.activation_func = activation_func

    def feedforward(self, x, h_prev):
        return self.activation_func.output(self.A.dot(x) + self.B.dot(h_prev))

    def forwardpropa(self, input_list, labels):
        hs = [0]
        ys = []
        cost = 0
        for t, data in enumerate(zip(input_list, labels)):
            x, label = data
            output = self.feedforward(x, hs[t])
            
            hs.append(output)
            ys.append(output)
            cost += self.cost_func.fn(x, label)
        return cost, ys, hs

    def backwardpropa(self, ys, hs, labels):
        for t in len(ys):
            dyt = (ys[t]-labels[t]) * self.activation_func.prime(ys[t])
        return 0
            

class QuadraticCost(object):

    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.

        """
        return 0.5*np.linalg.norm(a-y)**2

    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer."""
        return (a-y) * sigmoid_prime(z)

class tanh_func:
    @staticmethod
    def output(z):
        return (np.exp(z)-np.exp(-z)) / (np.exp(z)+np.exp(-z))

    @staticmethod
    def prime(z):
        return 1 - ((np.exp(z)-np.exp(-z)) / (np.exp(z)+np.exp(-z)))**2

if __name__=='__main__':
    input_vertor_size = 4
    output_vertor_size = 3
    x = [np.arange(1,input_vertor_size+1).reshape(4,1)+i for i in range(5)] # input vertor list
    y = [np.zeros((output_vertor_size,1))+i for i in range(5)] # correct answer list
    
    A = np.zeros((output_vertor_size, input_vertor_size))
    B = np.zeros((output_vertor_size, output_vertor_size))
    
    func1 = func(A, B, tanh_func, QuadraticCost)
    cost, ys, hs = func1.forwardpropa(x, y)
    print(ys)
    print(hs)
