"""Helper methods"""

def create_batches(inputs, labels, size):
    return (
        [inputs[i:i+size] for i in range(0, len(inputs), size)],
        [labels[i:i+size] for i in range(0, len(labels), size)]
    )


def train(model, criterion, inputs, labels, epochs=100, batch_size=100, gamma=0.0001):
    train_batches = create_batches(inputs, labels, batch_size)
    for epoch in range(epochs):
        losses = []
        for inputs, labels in zip(*train_batches):
            outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            losses.append(loss.item())
            # reset gradients
            model.grad_to_zero()
            
            # perform backward pass:
            # compute gradient of loss from its definition (last layer)
            grad_loss = criterion.backward()
            # ... and propagate derivatives backwards
            model.backward(grad_loss)
            for param in model.param():
                # SGD step -> adjust network parameters
                param -= gamma * param.grad
        
        avg_loss = sum(losses) / batch_size
        if epoch % 10 == 0:
            print('Epoch {}/{} - Average loss: {:.2f}'.format(str(epoch+1).zfill(3), epochs, avg_loss))
            
            
def compute_accuracy(model, inputs, labels):
    outputs = model.forward(inputs)
    predictions = outputs.round()
    accuracy = (predictions == labels).sum() / len(labels)
    return accuracy.item()

