import numpy as np
import matplotlib.pyplot as plt


class ExponentialDecay:
    def __init__(self, initAlpha=0.01, factor=0.25, dropEvery=10):
        # store the base initial learning rate, drop factor, and
        # epochs to drop every
        self.initAlpha = initAlpha
        self.factor = factor
        self.dropEvery = dropEvery

    def __call__(self, epoch):
        # compute learning rate for the current epoch
        exp = np.floor((1 + epoch) / self.dropEvery)
        alpha = self.initAlpha * (self.factor ** exp)

        # return the learning rate
        return float(alpha)
    
    def plot(self, epochs=100):
        # compute the set of learning rates for each corresponding
        # epoch
        alpha = [self(i) for i in range(0, epochs)]
        # plot the learning rates
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(np.arange(0, epochs), alpha)
        plt.title("Exponential Decay Learning Rate")
        plt.xlabel("Epoch #")
        plt.ylabel("Learning Rate")
        plt.show()
        
# initialize the exponential decay learning rate and plot the
# learning rates for 100 epochs
ed = ExponentialDecay()

ed.plot()
