"""Helper methods"""
import pickle
from typing import Any
from torch import Tensor, empty

import modules as m
from data import DataLoader

def optimize(
    model: m.Module,
    train_data: tuple[Tensor, Tensor],
    test_data: tuple[Tensor, Tensor],
    criterion: m.Module = m.LossMSE(),
    epochs: int = 100,
    batch_size: int = 100,
    lr: float = 0.001,
    verbose: Any = True
) -> tuple[Tensor]:
    """
    :param model: the NN model
    :param train_data: training data
    :param test_data: test data
    :param criterion: loss function
    :param epochs: number of epochs
    :param batch_size: training minibatch size
    :param lr: the learning rate
    :param verbose: whether to print progress
    :returns: tensors of losses (batch, training, test)
    """
    train_loader = DataLoader(train_data, batch_size=batch_size)
    model.reset_parameters()
    optimizer = m.OptimizerSGD(model.parameters(), lr)
    
    N = len(train_data[0])
    
    num_batches = N // batch_size
    batch_losses = empty(epochs * num_batches)
    test_losses = empty(epochs)
    train_losses = empty(epochs)
    
    for epoch in range(epochs):
        for batch_idx, (minibatch, labels) in enumerate(train_loader):
            # reset gradients
            optimizer.zero_grad()
            outputs = model(minibatch)
            
            loss = criterion(outputs, labels)
            
            # perform backward pass:
            # compute gradient of loss from its definition (last layer)
            grad_loss = criterion.backward()
            # ... and propagate derivatives backwards
            model.backward(grad_loss)
            
            # compute loss on entire train set after each minibatch
            batch_losses[epoch * num_batches + batch_idx] = criterion(model(train_data[0]), train_data[1])
            
            optimizer.step()
        
        # compute losses after each epoch
        train_loss = criterion(model(train_data[0]), train_data[1])
        test_loss = criterion(model(test_data[0]), test_data[1])
        train_losses[epoch] = train_loss
        test_losses[epoch] = test_loss
        if verbose:
            print(
                'Epoch {}/{} - Train loss: {:.2f} - Test loss: {:.2f}'.format(
                    str(epoch+1).zfill(3), epochs, train_loss, test_loss
                )
            )
    return batch_losses, train_losses, test_losses


def predict(model: m.Module, inputs: Tensor) -> Tensor:
    """
    :param model: the NN model
    :param inputs: input tensor
    :returns: binary predictions of the model given the inputs
    """
    # assuming outputs are in the range [0, 1]
    return model.forward(inputs).round()

            
def compute_accuracy(model: m.Module, inputs: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
    """
    :param model: the NN model
    :param inputs: input tensor
    :param labels: ground truth tensor
    :returns: total accuracy along with correctness tensor and predictions
    """
    predictions = predict(model, inputs)
    correct_class = predictions == labels
    accuracy = correct_class.sum() / len(labels)
    return accuracy, correct_class, predictions


def pickle_dump(filename, obj):
    with open(f'{filename}.pkl', 'wb') as file:
        pickle.dump(obj, file)

def pickle_load(filename):
    with open(f'{filename}.pkl', 'rb') as file:
        return pickle.load(file)
