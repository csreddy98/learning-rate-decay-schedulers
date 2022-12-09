
import matplotlib.pyplot as plt
import numpy as np

class StepLRDecay:
    def __init__(self, init_lr, decay_rate, decay_step):
        self.init_lr = init_lr
        self.decay_rate = decay_rate
        self.decay_step = decay_step

    def step(self, epoch):
        lr = self.init_lr * self.decay_rate ** (epoch // self.decay_step)
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
    decay_rate = 0.9
    decay_step = 10
    step_lr_decay = StepLRDecay(init_lr, decay_rate, decay_step)
    step_lr_decay.plot(epochs)