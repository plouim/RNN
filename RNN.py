import numpy as np

timestep = 5
hidden_layer_size = 10
input_layer_size = 4
output_layer_size = 4


class RNN:
    def __init__(self, input_layer_size, hidden_layer_size, output_layer_size, activation_func):
        self.input_layer_size = input_layer_size
        self.hidden_layer_size = hidden_layer_size
        self.output_layer_size = output_layer_size
        self.activation_func = activation_func
        
        self.Whx = np.zeros([hidden_layer_size, input_layer_size])
        self.Whh = np.zeros([hidden_layer_size, hidden_layer_size])
        self.Wyh = np.zeros([output_layer_size, hidden_layer_size])
        
        self.Bh = np.zeros([hidden_layer_size, 1])
        self.By = np.zeros([output_layer_size, 1])

    def output_h_cur(self, x, h_prev):
        z = self.Whx.dot(x) + self.Whh.dot(h_prev) + self.Bh
        return self.activation_func.output(z)
    
    def output_y(self, h_cur) 
        return self.Wyh.dot(h_cur) + self.By

    def feedforward(self, x, h_prev):
        h_cur = self.output_h_cur(x, h_prev)
        y = self.output_y(h_cur)
        return softmax.output(y)
    
    def forwardpropa(self, inputs, targets, h_prev):
        hs, ys, ps = [], [], [], []
        cost= 0

        hs[0] = np.copy(h_prev)
        hs[1] = self.output_h_cur(inputs[0], h_prev)
        ys[0] = self.output_y(h[0])
        ps[0] = softmax.output(ys[0])
        
        for t, x in enumertate(inputs, start=1):
            hs[t] = self.output_h_cur(x, hs[t-1]
            ys[t] = self.y(hs[t])
            ps[t] = softmax.output(ys[t])
            cost += -(np.array(target[t]) * np.log(x)).sum()

        return cost, ps, ys, hs

    def backwardpropa(self, ps, ys, hs, inputs, targets):
        dWhx, dWhh, dWyh = np.zeros_like(self.Whx), np.zeros_like(self.Whh), np.zeros_like(self.Wyh)
        dBh, dBy = np.zeros_like(self.Bh), np.zeros_like(self.By)
        dh_cur = np.zeros_like(hs[0])
        dC = [np.zeros_like(ps[0])]*len(inputs)
        dy = [np.zeros_like(ps[0])]*len(inputs)
        dz =  
        for t in range(len(inputs)):
            dC = -np.divide(targets, ps[t])
            dyt = dC * softmax.delta(ys[t])
            dht = self.Wyh.T.dot(dyt)
            dhraw = dht * tanh.delta(self.Whh.dot(hs[t-1])+self.Whx.dot(xs[t]+self.Bh))
            dh_prev = self.Whh.T.dot(dhraw)

            dWhh
            dW
            


            dC[t] = -np.divide(targets[t], ps[t])
            dy[t] = softmax.delta(ys[t])
            dBy = dy[t]
            dh[t] = self.Wyh.dot(dy[t])
            dWyh = dy[t].dot(hs[t])
            dz
            
        return 


    def learning(self, training_data, epochs, eta):
        
        





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
        return 
        
