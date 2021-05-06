from torch.random import manual_seed

import modules as m
from data import generate_data
from helpers import train, compute_accuracy

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

criterion = m.LossMSE()

train_inputs, train_labels, test_inputs, test_labels = generate_data()

train(model, criterion, train_inputs, train_labels)

train_accuracy = compute_accuracy(model, train_inputs, train_labels)
test_accuracy = compute_accuracy(model, test_inputs, test_labels)

print('##################################')
print('Train accuracy : {:.1f}%'.format(round(train_accuracy, 3) * 100))
print('Test accuracy  : {:.1f}%'.format(round(test_accuracy, 3) * 100))
