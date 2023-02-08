import numpy as np



class RNN:
    def __init__(self, input_layer_size, hidden_layer_size, output_layer_size, activation_func):
        self.input_layer_size = input_layer_size
        self.hidden_layer_size = hidden_layer_size
        self.output_layer_size = output_layer_size
        self.activation_func = activation_func
        self.Whx = np.random.randn(hidden_layer_size, input_layer_size)
        self.Whh = np.random.randn(hidden_layer_size, hidden_layer_size)
        self.Wyh = np.random.randn(output_layer_size, hidden_layer_size)
        
        self.Bh = np.random.randn(hidden_layer_size, 1)
        self.By = np.random.randn(output_layer_size, 1)

    def output_h_cur(self, x, h_prev):
        z = self.Whx.dot(x) + self.Whh.dot(h_prev) + self.Bh
        return self.activation_func.output(z)
    
    def output_y(self, h_cur):
        return self.Wyh.dot(h_cur) + self.By

    def softmax(self, y):
        return np.exp(y) / np.exp(y).sum()

    def feedforward(self, x, h_prev):
        h_cur = self.output_h_cur(x, h_prev)
        y = self.output_y(h_cur)
        return softmax.output(y)
   
    def forwardpropa(self, inputs, targets, h_prev):
        hs, ys, ps = [], [], []
        cost = 0

        hs.append(np.copy(h_prev))
        hs.append(self.output_h_cur(inputs[0], h_prev))
        ys.append(self.output_y(hs[0]))
        ps.append(self.softmax(ys[0]))
        cost = -(np.array(targets[0]) * np.log(ps[0])).sum()
        print("cost 1 : {}".format(cost))
        for t in range(1, len(inputs)):
            hs.append(self.output_h_cur(inputs[t], hs[t-1]))
            ys.append(self.output_y(hs[t]))
            ps.append(self.softmax(ys[t]))
            print("cost {} : {}".format(t+1, -(np.array(targets[t]) * np.log(ps[t])).sum()))
            cost += -(np.array(targets[t]) * np.log(ps[t])).sum() # CrossEntropy
        return cost, ps, ys, hs

    def backwardpropa(self, eta, ps, ys, hs, inputs, targets):
        
        # dC = [-1 for i in range(len(targets))]
        #dh_prev = np.zeros((hidden_layer_size, 1))
        dh_next = np.zeros((hidden_layer_size, 1))
        for t in reversed(range(len(inputs))):
            dyt = np.copy(ps[t])
            dyt[np.where(targets[t] == 1)] -= 1
            dWyh = np.dot(dyt, hs[t].T)
            dBy = dyt
            dht = np.dot(self.Wyh.T, dyt) + dh_next
            dhraw = (1 - hs[t]*hs[t]) * dht
            dBh = dhraw
            dWhx = np.dot(dhraw, inputs[t].T)
            dWhh = np.dot(dhraw, hs[t-1].T)
            dh_next = np.dot(self.Whh.T, dhraw)
            # hraw = self.Whx.dot(inputs[t]) + self.Whh.dot(hs[t-1]) + self.Bh

            # dyt = np.copy(ps[t])
            # dyt[np.where(targets[t] == 1)] -= 1
            # #dh_prev = dh_prev * tanh_func.prime(hraw)
            # dht = self.Wyh.T.dot(dyt)
            # dhraw = (dht + dh_prev) * tanh_func.prime(hraw)
            # # dh_prev = dh_prev * tanh_func.prime(hraw)
            
            # dWyh = dyt.dot(hs[t].T)
            # dWhh = dhraw.dot(hs[t-1].T) + dh_prev.dot(hs[t-1].T)
            # dWhx = dhraw.dot(inputs[t].T) + dh_prev.dot(inputs[t].T)

            # dBy = dyt
            # dBh = dhraw
            
            # dh_prev += self.Whh.T.dot(dhraw)
            
            self.Wyh -= eta * dWyh
            self.Whh -= eta * dWhh
            self.Whx -= eta * dWhx
            self.By -= eta * dBy
            self.Bh -= eta * dBh 

    def display_cost(self, inputs, targets):
        cost, _, _, _ = self.forwardpropa(inputs, targets, np.zeros((hidden_layer_size, 1))) 
        print("COST : {}".format(cost))

    def infer(self, inputs, h_prev):
        # h -> [1 0 0 0]
        # e -> [0 1 0 0]
        # l -> [0 0 1 0]
        # l -> [0 0 0 1]
        for x in inputs:
            output = self.feedforward(x, h_prev)
            print("INPUT: ", decode_vector(x),"| OUTPUT: {} ({})".format(decode_vector(output), output.max()))
            h_prev = self.output_h_cur(x, h_prev)
    def learning(self, eta, inputs, targets, h_0):
        cost, ps, ys, hs = net.forwardpropa(inputs, targets, h_0)
        print("CURRENT COST ", cost)
        net.backwardpropa(eta, ps, ys, hs, inputs, targets) 
        net.infer(inputs, h_0)
        net.display_cost(inputs, targets)



def decode_vector(x):
    match x.reshape(4,).argmax():
        case 0:
            return "h"
        case 1:
            return "e"
        case 2:
            return "l"
        case 3:
            return "o"

class tanh_func:
    @staticmethod
    def output(z):
        return (np.exp(z)-np.exp(-z)) / (np.exp(z)+np.exp(-z))

    @staticmethod
    def prime(z):
        return 1 - ((np.exp(z)-np.exp(-z)) / (np.exp(z)+np.exp(-z)))**2



class softmax:
    @staticmethod
    def output(z):
        return np.exp(z) / np.exp(z).sum()
    @staticmethod
    def delta(z):
        pass
        


if __name__=='__main__':
    timestep = 4
    input_layer_size = 4
    hidden_layer_size = 20
    output_layer_size = 4

    h = np.array([[1,0,0,0]]).reshape(input_layer_size, 1)
    e = np.array([[0,1,0,0]]).reshape(input_layer_size, 1)
    l = np.array([[0,0,1,0]]).reshape(input_layer_size, 1)
    o = np.array([[0,0,0,1]]).reshape(input_layer_size, 1)

    #h_0 = np.random.randn(hidden_layer_size, 1)
    h_0 = np.zeros((hidden_layer_size, 1))
    inputs  = [h, e, l, l]
    targets = [e, l, l, o]
    
    net = RNN(input_layer_size, hidden_layer_size, output_layer_size, tanh_func)
    cost, ps, ys, hs = net.forwardpropa(inputs, targets, h_0)
    print("INIT COST : ", cost)
    for i in range(1, 10):
        net.backwardpropa(0.01, ps, ys, hs, inputs, targets) 
        print("# {} iters".format(i))
        net.infer(inputs, h_0)
        net.display_cost(inputs, targets)
        print()
