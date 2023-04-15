import torch
import torch.optim as optim
import random
from mnist import load_data, train, evaluate, Net, HashedNet
from utils import get_equivalent_compression

use_cuda = torch.cuda.is_available()
torch.manual_seed(1)
random.seed(1)

device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
train_loader, valid_loader, test_loader = load_data(batch_size=50, kwargs=kwargs)

def train_nn(compress, hashed):
    input_dim = 784
    output_dim = 10
    
    if hashed:
        model = HashedNet(input_dim, output_dim, 1, 1000,
                          compress, dropout=0.25).to(device)
    else:
        eq_compress = get_equivalent_compression(input_dim, output_dim,
                                                 1000, 1, compress)
        model = Net(input_dim, output_dim, 1, 1000,
                    eq_compress, 0.25).to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.01,
                          momentum=0.9,
                          weight_decay=0.0)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                     factor=0.1,
                                                     patience=2,
                                                     verbose=True)

    print('The number of parameters is: {}'.format(
        sum(p.numel() for p in model.parameters() if p.requires_grad)))

    for epoch in range(1, 50 + 1):
        tr_loss = train(model, device, train_loader, optimizer, epoch, log_interval=50)
        val_loss, val_acc = evaluate(model, device, valid_loader)
        scheduler.step(val_loss)
        print('Epoch {} Train loss: {:.3f} Val loss: {:.3f} Val acc: {:.2f}%'.format(
              epoch, tr_loss, val_loss, val_acc))

    test_loss, test_acc = evaluate(model, device, test_loader)
    print('Test loss: {:.3f} Test acc: {:.2f}%'.format(test_loss, test_acc))
    
    return test_loss, test_acc

compression_rates = [1/64, 1/32, 1/16, 1/8, 1]
nn_records = []
hashednn_records = []
for compression in compression_rates:
    print("Compression rate: {}".format(compression))
    nn_records.append(train_nn(compression, hashed=False))
    hashednn_records.append(train_nn(compression, hashed=True))

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6,4))
ax.plot([100.0-x[1] for x in hashednn_records], 's-', linewidth=2, label='Hashed NN')
ax.plot([100.0-x[1] for x in nn_records], 'd--', linewidth=2, label='Equiv. NN')
ax.set_xticklabels(['1/64', '1/32', '1/16', '1/8', '1'])
ax.set_xticks(range(len(compression_rates)))
ax.set_xlabel('Compression ratio')
ax.set_ylabel('Error (%)')
plt.title("1 hidden layer, 1000 units")
plt.legend(handlelength=3)
plt.grid()
plt.savefig('example.svg')
plt.show()