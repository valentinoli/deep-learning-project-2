from __future__ import annotations
from typing import Optional, NoReturn, Literal
from torch import empty, Tensor
from math import sqrt

class Parameter():
    """Implements a module parameter"""
    def __init__(self, dim_out: int, dim_in: Optional[int] = None):
        dim = (dim_out, dim_in)
        if not dim_in:
            dim = dim_out
            
        self.data = empty(dim).zero_()
        self.grad = empty(dim).zero_()
#         self.dim_in = dim_in
#         self.dim = dim
#         self.reset()
        
    def __call__(self) -> Tensor:
        return self.data
        
    def __add__(self, other: Tensor) -> Parameter:
        self.data += other
        return self
        
    def __sub__(self, other: Tensor) -> Parameter:
        self.data -= other
        return self
    
#     def reset(self):
#         # parameter tensor
#         # control variance of derivatives of the loss
#         # so that weights evolve at the same rate across layers 
#         # -> Var[w] = 1/3N
#         bound = 1 / sqrt(self.dim_in)
#         self.data = empty(self.dim).uniform_(-bound, bound)
        
#         # parameter gradient tensor
#         self.grad = empty(self.dim).zero_()


class Module():
    """Superclass for framework modules"""
    def __call__(self, *inputs: tuple[Tensor]) -> Tensor:
        """Trigger forward pass when instance is called like a function"""
        return self.forward(*inputs)
    
    def forward(self, inputs: Tensor) -> Tensor:
        """Forward pass of the output"""
        self.inputs = inputs.clone()
        return inputs
        
    def backward(self, gradwrtoutput: Tensor) -> Tensor:
        """Backpropagation of the gradient"""
        return gradwrtoutput
        
    def parameters(self) -> list[Parameter]:
        """Returns the parameters of the module"""
        return []
    
    def reset_parameters(self, gain: float = 1) -> NoReturn:
        """Resets the parameters of the module"""
        pass

    
class ReLU(Module):
    """Rectified Linear Unit activation function"""
    gain = sqrt(2.0)
    
    def __init__(self):
        super().__init__()
        
    def forward(self, inputs):
        super().forward(inputs)
        return inputs.clamp(min=0)
    
    def backward(self, gradwrtoutput):
        # Hadamard product
        return gradwrtoutput * self.inputs.heaviside(empty(1).zero_())
        
    
class Tanh(Module):
    """Hyperbolic tangent activation function"""
    gain = 5.0 / 3
    
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        super().forward(inputs)
        return inputs.tanh()

    def backward(self, gradwrtoutput):
        # Hadamard product
        return gradwrtoutput * (1 - self.inputs.tanh() ** 2)


class Sigmoid(Module):
    """Sigmoid activation function"""
    gain = 1
    
    def __init__(self):
        super().__init__()
    
    def forward(self, inputs):
        super().forward(inputs)
        return inputs.sigmoid()
    
    def backward(self, gradwrtoutput):
        sigmoid = self.inputs.sigmoid()
        # Hadamard product
        return gradwrtoutput * sigmoid * (1 - sigmoid)


class Linear(Module):
    """Fully connected linear layer"""
    def __init__(
        self, 
        input_dim: int, 
        output_dim: Optional[int] = None, 
        reset_params: bool = False,
        init: Literal['default', 'xavier'] = 'default'
    ):
        super().__init__()
        # parameter initialization method
        self.init = init
        
        if not input_dim:
            raise ValueError('Argument input_dim is required')
            
        # set output dimension as the input dimension if it is not provided
        output_dim = output_dim if output_dim else input_dim

        # initialize
        self.weight = Parameter(output_dim, input_dim)
        self.bias = Parameter(output_dim)
        if reset_params:
            self.reset_parameters()

    def forward(self, inputs):
        super().forward(inputs)
        return inputs @ self.weight().T + self.bias()

    def backward(self, gradwrtoutput):
        self.weight.grad.add_(gradwrtoutput.T @ self.inputs)
        
        self.bias.grad.add_(gradwrtoutput.T.sum(1))
        return gradwrtoutput @ self.weight()
    
    def parameters(self):
        return [self.weight, self.bias]
    
    def init_xavier_normal(self, gain):
        # Glorot initialization
        # -> control variance of derivatives of the loss
        # so that weights evolve at the same rate across layers 
        std = gain * sqrt(2.0 / sum(self.weight().size()))
        self.bias().normal_(0, std)
        self.weight().normal_(0, std)
        
    def init_default(self):
        # Default initialization
        self.bias().normal_()
        self.weight().normal_()
    
    def reset_parameters(self, gain = 1):
        if self.init == 'xavier':
            self.init_xavier_normal()
        else:
            self.init_default()
        for p in self.parameters():
            p.grad.zero_()


class Sequential(Module):
    """A sequential container of modules, forming a neural net"""
    def __init__(self, *modules: tuple[Module]):
        super().__init__()
        # Modules are added to the container in the order they are passed in the constructor
        self.modules = modules
        self.reset_parameters()

    def forward(self, inputs):
        """Computes the full forward pass"""
        for module in self.modules:
            # Feed output of previous layer forward to the next
            inputs = module(inputs)
        return inputs

    def backward(self, gradwrtoutput):
        """Computes the full backward pass"""
        for module in reversed(self.modules):
            # Propagate backwards gradient of one layer to the previous
            gradwrtoutput = module.backward(gradwrtoutput)
        return gradwrtoutput

    def parameters(self):
        """Return list of parameters of all modules in order"""
        return [p for module in self.modules for p in module.parameters()]
    
    def reset_parameters(self):
        prev_module = None
        for module in self.modules:
            if prev_module and hasattr(prev_module, 'gain'):
                # previous module is an activation function with recommended gain
                module.reset_parameters(prev_module.gain)
            else:
                module.reset_parameters()
            prev_module = module



class LossMSE(Module):
    """Mean Squared Error"""
    def __init__(self):
        super().__init__()
        
    def forward(self, inputs, labels):
        super().forward(inputs)
        self.labels = labels
        return (inputs - labels).pow(2).sum() / len(inputs)
        
    def backward(self):
        return 2 * (self.inputs - self.labels) / len(self.inputs)

    
class LossBCE(Module):
    """Binary Cross Entropy"""
    def __init__(self):
        super().__init__()
        
    def forward(self, inputs, labels):
        super().forward(inputs)
        x = inputs
        y = labels
        self.labels = y
        loss = -(y.T @ x.log() + (1 - y).T @ (1 - x).log())
        return loss.sum() / len(y)
        
    def backward(self):
        x = self.inputs
        y = self.labels
        return ((1 - y) / (1 - x) - y / x) / len(y)

    
class OptimizerSGD():
    """Class for SGD optimization of model parameters"""
    def __init__(self, parameters: list[Parameter], lr: float):
        self.parameters = parameters
        self.learning_rate = lr
        
    def zero_grad(self) -> NoReturn:
        """Resets gradients of all parameters to zero"""
        for p in self.parameters:
            p.grad.zero_()
            
    def step(self) -> NoReturn:
        """Applies one step of SGD"""
        for param in self.parameters:
            # SGD step -> adjust network parameters
            param -= self.learning_rate * param.grad
            