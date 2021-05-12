from torch.random import manual_seed

import modules as m
from data import generate_data
from helpers import optimize, compute_accuracy, pickle_load, pickle_dump
from plot import plot_results

filename = 'model'

# Load data
manual_seed(42)
num_samples = 1000
train_data, test_data = generate_data(num_samples)

try:
    # Load trained model from memory
    print('Model loaded from pickle file')
    model = pickle_load(filename)
except FileNotFoundError:
    # Train the model

    input_dim = 2
    output_dim = 1
    nb_hidden = 25
    
    model = m.Sequential(
        m.Linear(input_dim, nb_hidden),
        m.Tanh(),
        m.Linear(nb_hidden),
        m.Tanh(),
        m.Linear(nb_hidden),
        m.Tanh(),
        m.Linear(nb_hidden, output_dim),
        m.Sigmoid()
    )

    epochs = 100
    batch_size = 50
    lr = 0.16

    print(f'Number of samples: {num_samples}')
    print(f'Epochs: {epochs}')
    print(f'Batch size: {batch_size}')
    print(f'Learning rate: {lr}')
    print('##################################')
    
    optimize(model, train_data, test_data, epochs=epochs, batch_size=batch_size, lr=lr)
    
    print('##################################')
    
    pickle_dump(filename, model)


accuracy, correct_class, _ = compute_accuracy(model, *test_data)

print('Test accuracy  : {:.1f}%'.format(round(accuracy.item(), 3) * 100))

print('Plotting results...')
plot_results(train_data, test_data, correct_class)
