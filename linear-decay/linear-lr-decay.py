import numpy as np
import matplotlib.pyplot as plt

class LinearLRDecay:
    def __init__(self, init_lr, final_lr, epochs):
        self.init_lr = init_lr
        self.final_lr = final_lr
        self.epochs = epochs
    
    def step(self, epoch):
        lr = self.final_lr + (self.init_lr - self.final_lr) * (1 - epoch / self.epochs)
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
    init_lr = 0.1
    final_lr = 0.01
    linear_lr_decay = LinearLRDecay(init_lr, final_lr, epochs)
    linear_lr_decay.plot(epochs)