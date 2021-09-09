import argparse
import os
from loaddata import CellPoseData
from Cellpose_2D_PyTorch import UpdatedCellpose, SizeModel
import torch
import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import tqdm

from cellpose_src.utils import diameters

parser = argparse.ArgumentParser()
parser.add_argument('--results-dir', help='Folder in which to save experiment results.')
parser.add_argument('--learning_rate', type=float)
parser.add_argument('--momentum', type=float)
parser.add_argument('--batch-size', type=int)
parser.add_argument('--epochs', type=int)
parser.add_argument('--do-3D', help='Whether or not to use 3D-Cellpose (Must use 3D volumes).',
                    action='store_true', default=False)
parser.add_argument('--from-3D', help='Whether the input training source data is 3D: assumes 2D if set to False.',
                    action='store_true', default=False)
parser.add_argument('--cellpose-pretrained',
                    help='Location of the generalized cellpose model to use for diameter estimation.')
parser.add_argument('--train-dataset', help='The directory(s) containing data to be used for training.', nargs='+')
args = parser.parse_args()

assert not os.path.exists(args.results_dir),\
    'Results folder currently exists; please specify new location to save results.'
os.mkdir(args.results_dir)
torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

gen_cellpose = UpdatedCellpose(channels=1, device=device)
gen_cellpose = torch.nn.DataParallel(gen_cellpose)
gen_cellpose.to(device)
gen_cellpose.load_state_dict(torch.load(args.cellpose_pretrained))
gen_cellpose.eval()

size_model = SizeModel().to(device)
optimizer = torch.optim.SGD(size_model.parameters(), lr=args.learning_rate, momentum=args.momentum)
loss_fn = torch.nn.MSELoss()

train_dataset = CellPoseData('Training', args.train_dataset, do_3D=args.do_3D, from_3D=args.from_3D)
train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)  # num_workers=num_workers

# Training network
train_losses = []
start_train = time.time()
print('Beginning size model network training.')
for e in range(1, args.epochs + 1):
    optimizer.zero_grad()
    size_model.train()
    for (batch_data, batch_labels) in tqdm(train_dl, desc='Epoch {}/{}'.format(e, args.epochs)):
        batch_data = batch_data.float().to(device)
        batch_diams = torch.tensor([]).to(device)
        for i in range(len(batch_data)):
            med, _ = diameters(batch_labels[i])
            med = torch.tensor([med]).float().to(device)
            batch_diams = torch.cat((batch_diams, med))
        styles = gen_cellpose(batch_data, style_only=True)
        batch_predictions = size_model(styles)
        train_loss = loss_fn(batch_predictions, batch_diams)
        train_losses.append(train_loss.item())
        train_loss.backward()
        optimizer.step()

torch.save(size_model.state_dict(), os.path.join(args.results_dir, 'size_model.pt'))
end_train = time.time()

step_size = args.epochs/len(train_losses)
train_steps = np.arange(step_size, args.epochs + step_size, step_size)
plt.figure()
plt.plot(train_steps, train_losses)
plt.title('Training Loss')
plt.savefig(os.path.join(args.results_dir, 'Training_Loss'))
plt.show()

# Evaluation
size_model.eval()
start_eval = time.time()
eval_losses = []
diams = []
predictions = []
for (batch_data, batch_labels) in tqdm(train_dl, desc='Testing'):
    batch_data = batch_data.float().to(device)
    batch_diams = torch.tensor([]).to(device)
    for i in range(len(batch_data)):
        med, _ = diameters(batch_labels[i])
        med = torch.tensor([med]).float().to(device)
        batch_diams = torch.cat((batch_diams, med))
    styles = gen_cellpose(batch_data, style_only=True)
    batch_predictions = size_model(styles)
    eval_loss = loss_fn(batch_predictions, batch_diams)
    eval_losses.append(eval_loss.item())
    diams.append(batch_diams.tolist())
    predictions.append(batch_predictions.tolist())

plt.figure()
plt.scatter(diams, predictions)
diam_range = (min(diams), max(diams))
plt.plot(diam_range, diam_range)
plt.title('True Diameters vs. Predicted')
plt.xlabel('True Diameters')
plt.ylabel('Predicted Diameters')
plt.savefig(os.path.join(args.results_dir, 'Diameter_Predictions'))
plt.show()

with open(os.path.join(args.results_dir, 'logfile.txt'), 'w') as txt:
    txt.write('Time to train: {}\n'.format(time.strftime("%H:%M:%S", time.gmtime(end_train - start_train))))
    txt.write('Learning rate: {}; Momentum: {}\n'.format(args.learning_rate, args.momentum))
    txt.write('Epochs: {}; Batch size: {}\n'.format(args.epochs, args.batch_size))
    txt.write('Loss: {}'.format(loss_fn))
    txt.write('Training dataset(s): {}\n'.format(args.train_dataset))
    txt.write('Cellpose model for size prediction: {}\n'.format(args.cellpose_pretrained))
