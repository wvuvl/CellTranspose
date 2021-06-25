import os
from loaddata import StandardizedTiffData
from transforms import Reformat, Normalize1stTo99th
from Cellpose_2D_PyTorch import UpdatedCellpose, SizeModel
import torch
import torchvision
import numpy as np
from time import time
import matplotlib.pyplot as plt

from cellpose_src.utils import diameters
from misc_utils import elapsed_time

# style has shape (1, 256)
results_dir = '/home/mrkeaton/Documents/Datasets/Neuro_Proj1_Data/2D Toy Dataset - 3-dim/results/size_model_results/results3'
assert not os.path.exists(results_dir), 'Results folder currently exists; please specify new location to save results.'
os.mkdir(results_dir)
torch.cuda.empty_cache()
learning_rate = 0.001
momentum = 0
batch_size = 1
epochs = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

gen_cellpose = UpdatedCellpose(channels=1).to(device)
gen_cellpose.load_state_dict(torch.load('/home/mrkeaton/Documents/Datasets/Neuro_Proj1_Data/'
                                        '2D Toy Dataset - 3-dim/results/results102/trained_model.pt'))
gen_cellpose.eval()

data_transform = torchvision.transforms.Compose([
    Reformat(),
    Normalize1stTo99th()
])
label_transform = torchvision.transforms.Compose([
    Reformat()
])

train_dataset = StandardizedTiffData('/home/mrkeaton/Documents/Datasets/Neuro_Proj1_Data/2D Toy Dataset - 3-dim',
                                     train=True, do_3D=False, d_transform=data_transform, l_transform=label_transform)
train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # num_workers=num_workers

size_model = SizeModel().to(device)
optimizer = torch.optim.SGD(size_model.parameters(), lr=learning_rate, momentum=momentum)
loss_fn = torch.nn.MSELoss()


# Training network
train_losses = []
print('Beginning size model network training.')
for e in range(1, epochs + 1):
    print('Epoch {}/{}:'.format(e, epochs))
    optimizer.zero_grad()
    size_model.train()
    start_train = time()
    # for (batch_data, batch_labels, _) in tqdm(data_loader):
    for (batch_data, batch_labels, _) in train_dl:
        batch_data = batch_data.float().to(device)
        batch_diams = torch.tensor([]).to(device)
        for i in range(len(batch_data)):
            med, _ = diameters(batch_labels[i])
            med = torch.tensor([med]).float().to(device)
            batch_diams = torch.cat((batch_diams, med))
        styles = gen_cellpose.style_forward(batch_data)
        batch_predictions = size_model(styles)
        train_loss = loss_fn(batch_predictions, batch_diams)
        train_losses.append(train_loss.item())
        train_loss.backward()
        optimizer.step()
    print('Train time: {}'.format(elapsed_time(time() - start_train)))

torch.save(size_model.state_dict(), os.path.join(results_dir, 'size_model.pt'))

with open(os.path.join(results_dir, 'settings.txt'), 'w') as txt:
    txt.write('Learning rate: {}; Momentum: {}\n'.format(learning_rate, momentum))
    txt.write('Epochs: {}; Batch size: {}'.format(epochs, batch_size))

# train_steps = np.linspace(0, epochs, len(losses), endpoint=False)
step_size = epochs/len(train_losses)
train_steps = np.arange(step_size, epochs + step_size, step_size)
plt.figure()
plt.plot(train_steps, train_losses)
plt.title('Training Loss')
plt.savefig(os.path.join(results_dir, 'Training_Loss'))

# Evaluation
size_model.eval()
print('Evaluating network.')
start_eval = time()
eval_losses = []
diams = []
predictions = []
for (batch_data, batch_labels, _) in train_dl:
    batch_data = batch_data.float().to(device)
    batch_diams = torch.tensor([]).to(device)
    for i in range(len(batch_data)):
        med, _ = diameters(batch_labels[i])
        med = torch.tensor([med]).float().to(device)
        batch_diams = torch.cat((batch_diams, med))
    styles = gen_cellpose.style_forward(batch_data)
    batch_predictions = size_model(styles)
    eval_loss = loss_fn(batch_predictions, batch_diams)
    eval_losses.append(eval_loss.item())
    diams.append(batch_diams.tolist())
    predictions.append(batch_predictions.tolist())

print('Evaluation time: {}'.format(elapsed_time(time() - start_eval)))

plt.figure()
plt.scatter(diams, predictions)
plt.title('True Diameters vs. Predicted')
plt.savefig(os.path.join(results_dir, 'Diameter_Predictions'))
