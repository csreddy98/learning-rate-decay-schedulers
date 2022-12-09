# Learning Rate Scheduler
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/csreddy98/learning-rate-decay-schedulers/blob/fdccd94e6f540c01582fe1f409a1f6ef70633e69/LICENSE)
[]


This repository contains a collection of custom learning rate schedulers. These Scheduler classes can be used to decay the learning rate during training.

## Installation

```bash
pip install numpy matplotlib
```


## Available Schedulers

* [Linear Decay](https://github.com/csreddy98/learning-rate-decay-schedulers/blob/fdccd94e6f540c01582fe1f409a1f6ef70633e69/linear-decay/linear-lr-decay.py)
* [Exponential Decay](https://github.com/csreddy98/learning-rate-decay-schedulers/blob/main/exponential-decay/exponential-lr-decay.py)
* [Step Decay](https://github.com/csreddy98/learning-rate-decay-schedulers/blob/main/step-decay/step-lr-decay.py)
* [Polynomial Decay](https://github.com/csreddy98/learning-rate-decay-schedulers/blob/main/step-decay/step-lr-decay.py)

----
## Explanation

### Linear Decay Scheduler: 
The learning rate is decayed linearly from the initial learning rate to minimum learning rate over a specified number of epochs. The formula for the learning rate is given below:
```
lr = lr_min + (lr_max - lr_min) * (1 - epoch / epochs)
```
### Exponential Decay Scheduler:

The learning rate is decayed exponentially over a specified number of epochs. The formula for the learning rate is given below:
```
lr = lr_min + (lr_max - lr_min) * (decay_rate) ^ (epoch / epochs)
```

### Step Decay Scheduler:

The learning rate is decayed by a factor of gamma every step_size epochs. The formula for the learning rate is given below:
```
lr = lr_max * gamma ^ (epoch / step_size)
```
### Polynomial Decay Scheduler:

The learning rate is decayed by a factor of gamma every step_size epochs. The formula for the learning rate is given below:
```
lr = lr_max * (1 - epoch / epochs) ^ (power)
```
---


## TODO

* [ ] Add more schedulers
* [ ] Create examples for each scheduler
* [ ] Add documentation
---
## Contributing
Everyone is welcome to contribute to this repository. 


