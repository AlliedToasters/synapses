from distutils.core import setup

long_desc = """
A PyTorch implementation of Scalable Training of Artificial Neural Networks with Adaptive Sparse Connectivity inspired by Network Science by Mocanu et al. (https://arxiv.org/abs/1707.04780)
Uses sparse data structures. Not super fast yet, but less memory-intensive than the masked dense weight matrices used
in the proof-of-concept code released with the paper.
"""

setup(
    name='synapses',
    version='0.0.14',
    description='Adaptive Sparse Connectivity for Neural Networks in PyTorch',
    long_description=long_desc,
    author='Michael Klear',
    author_email='michael.r.klear@gmail.com',
    url='https://github.com/AlliedToasters/synapses/archive/v0.0.14.tar.gz',
    install_requires=['torch', 'torch-scatter'],
    packages=['synapses']
)
