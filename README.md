# Synapses
## A Pytorch Implementation of [Sparse Evolutionary Training (SET)](https://arxiv.org/abs/1707.04780) for Neural Networks
Based on [research](https://www.nature.com/articles/s41467-018-04316-3) published by Mocanu et al., "Scalable Training of Artificial Neural Networks with Adaptive Sparse Connectivity inspired by Network Science"<br><br>

This software is intended to be a starting point for developing more efficient implementations of SET networks. I am not a computer scientist; I have not gotten anywhere close to the potential computational efficiency of this training/inference procedure. Synapses uses a truly sparse weight matrix and transformations (not just a masked dense/fully-connected layer as used in the proof-of-concept work by the authors).<br><br>

## Features
Synapses v0.0.1x offers the following:<br>
 - Sparse weight matrices & transformations
 - An API for rapid implementation and experimentation with SET using PyTorch<br>
 
My hope is that SET will gain popularity and this project will rapidly improve through community support.<br><br>

Synapses is built entirely on PyTorch using pytorch v0.4.1; it probably works with other versions but has not been tested.<br><br>

To use, install pytorch and install synapses with:<br>

`
pip install synapses
`

for a usage demonstration, take a look at the [MNIST example notebook](MNIST_demo.ipynb).

## Note about Optimizers
Synapses recycles parameters after resetting connections; many optimizers (SGD with momentum, RMSprop, adaptive learning rate methods) use a buffer with memory about previous steps to compute weight updates. It's important to treat these buffers properly when re-initializing parameters. Synapses will take care of this, but to do so you must pass your optimizer object to your SETLayers after initializing the optimizer. Simply do (something like):

```
my_optimizer = torch.optim.SGD(params=model.parameters(), lr=1e-3, momentum=.9)
layer.optimizer = my_optimizer
```
With vanilla SGD, this step is not necessary.

TODO:<br>
 - Fix bug with optimizers: Only vanilla SGD is working right; need to reset momentum buffers when resetting connections.
 - Build unit testing script that includes time benchmarking
 - Improve computational efficiency
