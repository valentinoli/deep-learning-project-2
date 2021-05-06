from __future__ import annotations
from typing import Optional, NoReturn
from torch import empty, Tensor

class Parameter():
    """Implements a module parameter"""
    def __init__(self, dim_1: int, dim_2: Optional[int] = None):
        dim = (dim_1, dim_2)
        if not dim_2:
            dim = dim_1
        # parameter tensor
        self.data = empty(dim).normal_()
        # parameter gradient tensor
        self.grad = empty(self.data.size()).zero_()
        
    def __call__(self) -> Tensor:
        return self.data
        
    def __add__(self, other: Tensor) -> Parameter:
        self.data += other
        return self
        
    def __sub__(self, other: Tensor) -> Parameter:
        self.data -= other
        return self


class Module():
    """Superclass for framework modules"""
    def __call__(self, *inputs: tuple[Tensor]) -> Tensor:
        """Trigger forward pass when instance is called like a function"""
        return self.forward(*inputs)
    
    def forward(self, inputs: Tensor) -> NoReturn:
        """Forward pass of the output"""
        self.inputs = inputs.clone()
        
    def backward(self, gradwrtoutput: Tensor) -> Tensor:
        """Backpropagation of the gradient"""
        raise NotImplementedError
        
    def param(self) -> list[Parameter]:
        """Returns the parameters of the module"""
        return []
    
    def grad_to_zero(self) -> NoReturn:
        """Resets all gradients of the module to zero"""
        pass

    
class ReLU(Module):
    """Rectified Linear Unit activation function"""
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
    def __init__(self, input_dim: int, output_dim: Optional[int] = None):
        super().__init__()
        if not input_dim:
            raise ValueError('Argument input_dim is required')
            
        # set output dimension as the input dimension if it is not provided
        output_dim = output_dim if output_dim else input_dim

        # initialize
        self.weight = Parameter(output_dim, input_dim)
        self.bias = Parameter(output_dim)

    def forward(self, inputs):
        super().forward(inputs)
        return inputs @ self.weight().T + self.bias()

    def backward(self, gradwrtoutput):
        self.weight.grad.add_(gradwrtoutput.T @ self.inputs)
        
        self.bias.grad.add_(gradwrtoutput.T.sum(1))
        return gradwrtoutput @ self.weight()
    
    def param(self):
        return [self.weight, self.bias]
    
    def grad_to_zero(self):
        self.weight.grad.zero_()
        self.bias.grad.zero_()



class Sequential(Module):
    """A sequential container of modules, forming a neural net"""
    def __init__(self, *modules: tuple[Module]):
        super().__init__()
        # Modules are added to the container in the order they are passed in the constructor
        self.modules = modules

    def forward(self, inputs):
        """Computes the full forward pass"""
        output = inputs
        for module in self.modules:
            # Feed output of previous layer forward to the next
            output = module(output)
        return output

    def backward(self, gradwrtoutput):
        """Computes the full backward pass"""
        grad = gradwrtoutput
        for module in reversed(self.modules):
            # Propagate backwards gradient of one layer to the previous
            grad = module.backward(grad)
        return grad

    def param(self):
        """Return list of parameters of all modules in order"""
        return [p for module in self.modules for p in module.param()]
    
    def grad_to_zero(self):
        """Set gradient of all parameters in the network to zero"""
        for module in self.modules:
            module.grad_to_zero()
    



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
