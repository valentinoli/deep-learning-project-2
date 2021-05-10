from torch.random import manual_seed

import modules as m
from data import generate_data
from helpers import optimize, compute_accuracy

manual_seed(42)

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

num_samples=1000
epochs=100
batch_size=50
lr=0.01

train_data, test_data = generate_data(num_samples)

optimize(model, train_data, test_data, epochs=epochs, batch_size=batch_size, lr=lr)

train_accuracy = compute_accuracy(model, *train_data)
test_accuracy = compute_accuracy(model, *test_data)

print('##################################')
print('Train accuracy : {:.1f}%'.format(round(train_accuracy, 3) * 100))
print('Test accuracy  : {:.1f}%'.format(round(test_accuracy, 3) * 100))
