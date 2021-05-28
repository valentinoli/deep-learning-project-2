import time, sys

from torch import set_grad_enabled

import lamp as l
from data import generate_data
from helpers import optimize, compute_accuracy, pickle_load, pickle_dump
from plot import plot_results

set_grad_enabled(False)

filename = 'model'

# Load data
num_samples = 1000
train_data, test_data = generate_data(num_samples)

try:
    # Load trained model from pickle
    model = pickle_load(filename)
    print(f'Model loaded from file {filename}.pkl')
except FileNotFoundError:
    # Train the model

    input_dim = 2
    output_dim = 1
    nb_hidden = 25
    
    model = l.Sequential(
        l.Linear(input_dim, nb_hidden),
        l.Tanh(),
        l.Linear(nb_hidden),
        l.Tanh(),
        l.Linear(nb_hidden),
        l.Tanh(),
        l.Linear(nb_hidden, output_dim),
        l.Sigmoid()
    )

    epochs = 100
    batch_size = 50
    lr = 0.16
    shuffle = False

    print(f'Number of samples: {num_samples}')
    print(f'Epochs: {epochs}')
    print(f'Batch size: {batch_size}')
    print(f'Learning rate: {lr}')
    print(f'Shuffle before each epoch: {shuffle}')
    print()
    
    for i in reversed(range(1, 4)):
        sys.stdout.write("\rStarting in %i" % i)
        sys.stdout.flush()
        time.sleep(1)

    sys.stdout.write('\r##################################\n')
    
    optimize(model, train_data, test_data, epochs=epochs, batch_size=batch_size, lr=lr, shuffle=shuffle)
    
    print('##################################')
    
    # uncomment the following lines to enable model pickling
    # pickle_dump(filename, model)
    # print(f'Trained model pickled to {filename}.pkl')


accuracy, correct_class, _ = compute_accuracy(model, *test_data)

print('Test accuracy  : {:.1f}%'.format(round(accuracy.item(), 3) * 100))

print('Plotting data and results...')
plot_results(train_data, test_data, correct_class)
