"""Helper methods"""
import modules as m
from data import DataLoader
from torch import Tensor
from typing import Any

def optimize(
    model: m.Module,
    train_data: tuple[Tensor, Tensor],
    test_data: tuple[Tensor, Tensor],
    criterion: m.Module = m.LossMSE(),
    epochs: int = 100,
    batch_size: int = 100,
    lr: float = 0.001,
    verbose: Any = True
) -> list[float]:
    """
    :param model: the NN model
    :param train_data: training data
    :param test_data: test data
    :param criterion: loss function
    :param epochs: number of epochs
    :param batch_size: training minibatch size
    :param lr: the learning rate
    :param verbose: whether to print progress
    """
    train_loader = DataLoader(train_data, batch_size=batch_size)
    test_inputs, test_labels = test_data
    
    optimizer = m.OptimizerSGD(model.parameters(), lr)
    losses = []
    
    for epoch in range(epochs):
        for minibatch, labels in train_loader:
            outputs = model(minibatch)
            
            loss = criterion(outputs, labels)
            
            # reset gradients
            optimizer.zero_grad()
            
            # perform backward pass:
            # compute gradient of loss from its definition (last layer)
            grad_loss = criterion.backward()
            # ... and propagate derivatives backwards
            model.backward(grad_loss)
            
            optimizer.step()
        
        # compute test loss
        test_loss = criterion(model(test_inputs), test_labels)
        losses.append(test_loss.item())
        if verbose:
            print(
                'Epoch {}/{} - Test loss: {:.2f}'.format(
                    str(epoch+1).zfill(3), epochs, test_loss
                )
            )
    return losses


def predict(model: m.Module, inputs: Tensor) -> Tensor:
    """
    :param model: the NN model
    :param inputs: input tensor
    :returns: binary predictions of the model given the inputs
    """
    # assuming outputs are in the range [0, 1]
    return model.forward(inputs).round()

            
def compute_accuracy(model: m.Module, inputs: Tensor, labels: Tensor) -> float:
    """
    :param model: the NN model
    :param inputs: input tensor
    :param labels: ground truth tensor
    :returns: total accuracy of the predictions
    """
    predictions = predict(model, inputs)
    accuracy = (predictions == labels).sum() / len(labels)
    return accuracy.item()
