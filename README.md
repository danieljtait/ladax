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
The motivation for this distinction is that it often
easier to canonicalise the parameters of a distribution 
returned by a layer, and outsource subtleties and
variations of these parameterisations in a seperate
module.

The following code snippet violates the three definitions
above (WIP!), but gives an idea
```python 
class SVGP(nn.Module):
    def apply(self, x):
        kernel_fn = kernel_provider(x, **kernel_fn_kwargs)
        inducing_var = inducing_variable_provider(x, kernel_fn, **inducing_var_kwargs)
        vgp = SVGPLayer(x, mean_fn, kernel_fn, inducing_var)
        return vgp
```
in the above we have the following 
* A `GP` is canonicalised by a `mean_fn` and `kernel_fn`, we abstract away the
specification and parameterisation of these objects to another module.

## ToDo

* Remove `likelihoods` and put this functionality into `losses`, and make the
layer loss functions in `losses` import and parameterise the objects in
`distributions`. 
* Kernel algebra -- sums, products of kernels etc.
* Apply kernel providers only to slices of index points
* Examples of deep GPs with multiple GPs per layer, perhaps create an `IndependentGP`
collection
* More general multioutput GPs
* Stop putting `index_points` through the kernel provider layers, just pass the number
of features 
* More losses -- Poisson etc. for count data