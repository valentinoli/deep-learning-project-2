"""Helper methods"""
from modules import OptimizerSGD, LossMSE
from data import DataLoader


def optimize(model, train_data, test_data, criterion=LossMSE(), epochs=100, batch_size=100, lr=0.001):
    train_loader = DataLoader(train_data, batch_size=batch_size)
    test_inputs, test_labels = test_data
    
    optimizer = OptimizerSGD(model.parameters(), lr)
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
        test_loss = criterion(model(test_inputs), test_labels).item()
        losses.append(test_loss)
        
        print(
            'Epoch {}/{} - Test loss: {:.2f}'.format(
                str(epoch+1).zfill(3), epochs, test_loss
            )
        )
    return losses
            
            
def compute_accuracy(model, inputs, labels):
    outputs = model.forward(inputs)
    # assuming outputs are in the range [0, 1]
    predictions = outputs.round()
    accuracy = (predictions == labels).sum() / len(labels)
    return accuracy.item()

