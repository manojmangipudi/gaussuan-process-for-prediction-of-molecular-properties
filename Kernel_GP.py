"""
Kernel is the important aspect of Gaussian processes, 
Tanimoto kernel is proven and showed ability in estimating molecular proprty prediction
"tanimoto similarity refers to the similarity of chemical elements, molecules or chemical 
compounds with respect to either structural or functional qualities" Soruce WIKI.

 IMPLEMENTATION:
 This kernel is implemented completely using Gpytorch library,
 Gpytorch library is exclusively designed for Gaussian Processes and Bayesian Optimization.
"""

# =================Kerenl implementation=========================================
"""
Kernel is the parameter that defines relationship between the points.
The kernel is implemented using gpytorch library.
"""
def broadcasting_elementwise(op, a, b):
    
    # Apply binary operation `op` to every pair in tensors `a` and `b`.

    # :param op: binary operator on tensors, e.g. torch.add, torch.substract
    # :param a: torch.Tensor, shape [n_1, ..., n_a]
    # :param b: torch.Tensor, shape [m_1, ..., m_b]
    # :return: torch.Tensor, shape [n_1, ..., n_a, m_1, ..., m_b]
    
    flatres = op(torch.reshape(a, [-1, 1]), torch.reshape(b, [1, -1]))
    return flatres


# import positivity constaint
from gpytorch.constraints import Positive

class TanimotoKernel(gpytorch.kernels.Kernel):
    # the tanimoto kernel is stationary
    is_stationary = True
    
    # we register the parameter when initializing the kernel
    def __init__(self, num_dimensions = None, offset_prior=None, variance_prior = None, variance_constraint = None, **kwargs):
        super().__init__(**kwargs)
        
        # registe the raw parameter
        self.register_parameter(name="raw_variance", parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1)))
   
        # set the parameter contraint to be positive, when nothing is specified
        if variance_constraint is None:
            variance_constraint = Positive()
            
        # register the constraint
        self.register_constraint("raw_variance", variance_constraint)
        
        #set the parameter prior
        if variance_prior is not None:            
            self.register_prior("variance_prior", variance_prior, lambda m: m.variance, lambda m, v: m._set_variance(v))
        
        if num_dimensions is not None:
            # Remove after 1.0
            warnings.warn("The `num_dimensions` argument is deprecated and no longer used.", DeprecationWarning)
            self.register_parameter(name="offset", parameter=torch.nn.Parameter(torch.zeros(1, 1, num_dimensions)))
        
        if offset_prior is not None:
            # Remove after 1.0
            warnings.warn("The `offset_prior` argument is deprecated and no longer used.", DeprecationWarning)
        self.register_parameter(name="raw_variance", parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1)))
    
      # now set the actual parameter
    @property
    def variance(self):
        return self.raw_variance_constraint.transform(self.raw_variance)
        
    @variance.setter
    def variance(self, value):
        return self._set_variance(value)
        
    def _set_variance(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_variance)
        # transform the actual value to a raw one by applying inverse transform
        self.initialize(raw_variance = self.raw_length_constraint.inverse_transform(value))            
    
    # define the kernel function
    def forward(self, X, X2 = None, last_dim_is_batch=False):
    
        if X2 is None:
            X2 = X
        Xs = torch.sum(torch.square(X), dim = -1)
        X2s = torch.sum(torch.square(X2), dim = -1)
        cross_product = torch.tensordot(X, X2, ([-1], [-1]))
    
        denominator = -cross_product + broadcasting_elementwise(torch.add, Xs, X2s)
    
        return self.variance * cross_product / denominator
