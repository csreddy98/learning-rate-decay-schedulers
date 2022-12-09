import numpy as np
import matplotlib.pyplot as plt


class PolynomialLRDecay:
    def __init__(self, init_lr, power, max_epochs):
        """
        :param init_lr: initial learning rate (float)
        :param power: power of polynomial (float)
        :param max_epochs: number of epochs (int)
        
        """
        self.init_lr = init_lr
        self.power = power
        self.max_epochs = max_epochs

    def step(self, epoch):
        """
        :param epoch: current epoch (int)
        :return: learning rate (float)
        """
        lr = self.init_lr * (1 - epoch / self.max_epochs) ** self.power
        return lr

    def plot(self, epochs):
        lrs = []
        for epoch in range(epochs):
            lr = self.step(epoch)
            lrs.append(lr)
        plt.plot(lrs)
        plt.xlabel("Epochs")
        plt.ylabel("Learning Rate")
        plt.show()


if __name__ == "__main__":
    epochs = 100
    init_lr = 0.01
    power = 2
    optimizer = None
    polynomial_lr_decay = PolynomialLRDecay(optimizer, init_lr, power, epochs)
    polynomial_lr_decay.plot(epochs)
