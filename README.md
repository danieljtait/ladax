# LADAX: Layers of distributions using FLAX/JAX

## Introduction

Small demonstration of using the FLAX package to create layers
of distributions. Current demonstrations focus on using Gaussian 
processes. Why? Because once the work is done in creating the
basic `GaussianProcessLayers` etc. we can use the FLAX functional
layers API to
1. Easily combine simpler GPs to create DeepGPs
2. Easily slot GPs into other deep learning frameworks. 


Briefly the design envisions three components

#### 1. Distributions
A probability distribution, represented as a valid JAX type,
this can be achieved by registering the object as a pytree
node. This process is made convenient using the 
`struct.dataclass` decorator from FLAX. 

#### 2. Distribution layers
These are instances of a `flax.nn.Module` objects which 
accept some input valid JAX type, and return an output 
in the form of a distribution.

#### 3. Providers
Like the above, only without an input! An example is 
the `RBFKernelProvider` which returns a `Kernel`, 
a `struct` decorated container of the exponentiated 
quadratic kernel function. Because these components
subclass `flax.nn.Module` they are a convenient place
to handle initialisation and storage of parameters.
