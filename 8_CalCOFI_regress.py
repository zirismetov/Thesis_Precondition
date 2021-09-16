import numpy as np, pandas as pd
import sklearn.metrics
import torch
import torch.utils.data
from sklearn.model_selection import train_test_split
import argparse
import time
from datetime import datetime
from taskgen_files import csv_utils_2, args_utils, file_utils

parser = argparse.ArgumentParser(description="Weather hypermarkets")
parser.add_argument('-sequence_name', type=str, default='sequence')
parser.add_argument('-run_name', type=str, default=str(time.time()))
parser.add_argument('-lr', type=float, default=1e-2)
parser.add_argument('-batch_size', type=int, default=32)
parser.add_argument('-test_size', type=float, default=0.20)
parser.add_argument('-epoch', type=int, default=10)
parser.add_argument('-dropout_mode', type=str, default='False')
parser.add_argument('-data_file_path', type=str, default='/Users/zafarzhonirismetov/Desktop/Work/CalCOFI/bottle.csv')

args, args_other = parser.parse_known_args()
args = args_utils.ArgsUtils.add_other_args(args, args_other)
args.sequence_name_orig = args.sequence_name
args.sequence_name += ('-' + datetime.utcnow().strftime(f'%y-%m-%d-%H-%M-%S'))

data_raw = pd.read_csv(args.data_file_path, low_memory=False)

# Predict temperature of water 1 features: salinity
data_raw = data_raw[['Salnty','T_degC']]
data_raw['Salnty'].replace(0, np.nan, inplace=True)
data_raw['T_degC'].replace(0, np.nan, inplace=True)
data_raw.fillna(method='pad', inplace=True)

class DatasetLoadColCOFI(torch.utils.data.Dataset):

    def __init__(self):
        self.X = data_raw['Salnty'].to_numpy()
        np_x = np.copy(self.X)
        self.X[:] = ((np_x[:] - np.min(np_x[:])) / (np.max(np_x[:]) - np.min(np_x[:])))
        self.X = np.expand_dims(self.X, axis=1)

        self.y = data_raw['T_degC'].to_numpy()
        np_y = np.copy(self.y)
        self.y[:] = ((np_y[:] - np.min(np_y[:])) / (np.max(np_y[:]) - np.min(np_y[:])))
        self.y = np.expand_dims(self.y, axis=1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, item):
        return self.X[item], self.y[item]


dataset = DatasetLoadColCOFI()

tr_idx = np.arange(len(dataset))

subset_train_data, subset_test_data = train_test_split(
    tr_idx,
    test_size=args.test_size,
    random_state=0)

dataset_train = torch.utils.data.Subset(dataset, subset_train_data)
dataset_test = torch.utils.data.Subset(dataset, subset_test_data)

dataloader_train = torch.utils.data.DataLoader(dataset_train,
                                               batch_size=args.batch_size,
                                               shuffle=True)

dataloader_test = torch.utils.data.DataLoader(dataset_test,
                                              batch_size=args.batch_size,
                                              shuffle=False)


class Model(torch.nn.Module):
    def __init__(self, dropout_mode):
        super(Model, self).__init__()

        if dropout_mode == 'True':
            prob = 0.5
        else:
            prob = 0

        self.layers = torch.nn.Sequential(
            torch.nn.Linear(in_features=1, out_features=256),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(p=prob),
            torch.nn.Linear(in_features=256, out_features=256),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(p=prob),
            torch.nn.Linear(in_features=256, out_features=64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(in_features=64, out_features=1),
        )

    def forward(self, x):
        y_prim = self.layers.forward(x)
        return y_prim


path_sequence = f'./results/{args.sequence_name}'
args.run_name += ('-' + datetime.utcnow().strftime(f'%y-%m-%d--%H-%M-%S'))
path_run = f'./results/{args.sequence_name}/{args.run_name}'
file_utils.FileUtils.createDir(path_run)
file_utils.FileUtils.writeJSON(f'{path_run}/args.json', args.__dict__)
csv_utils_2.CsvUtils2.create_global(path_sequence)
csv_utils_2.CsvUtils2.create_local(path_sequence, args.run_name)

model = Model(dropout_mode=args.dropout_mode)
opt = torch.optim.Adam(
    model.parameters(),
    lr=args.lr
)

losses_train = []
losses_test = []
R2_train = []
R2_test = []
metrics_mean_dict = {'loss_train': 0,
                     'R^2_train': 0,
                     'loss_test': 0,
                     'R^2_test': 0
                     }
for epoch in range(args.epoch):

    for dataloader in [dataloader_train, dataloader_test]:
        losses = []
        R2_s = []
        if dataloader is dataloader_test:
            model.eval()
            mode = 'test'
        else:
            model.train()
            mode = 'train'

        metrics_mean_dict[f'loss_{mode}'] = []
        metrics_mean_dict[f'R^2_{mode}'] = []

        for x, y in dataloader:

            y_prim = model.forward(torch.FloatTensor(x.float()))
            y = torch.FloatTensor(y.float())

            loss = torch.mean(torch.abs(y - y_prim))
            R2 = sklearn.metrics.r2_score(y.detach(), y_prim.detach())

            metrics_mean_dict[f'loss_{mode}'].append(loss)
            metrics_mean_dict[f'R^2_{mode}'].append(R2)

            losses.append(loss.item())
            R2_s.append(R2.item())

            if dataloader is dataloader_train:
                loss.backward()
                opt.step()
                opt.zero_grad()

        metrics_mean_dict[f'loss_{mode}'] = round((torch.mean(torch.FloatTensor(metrics_mean_dict[f'loss_{mode}'])))
                                                  .numpy().item(), 4)
        metrics_mean_dict[f'R^2_{mode}'] = round((torch.mean(torch.FloatTensor(metrics_mean_dict[f'R^2_{mode}'])))
                                                 .numpy().item(), 4)
        csv_utils_2.CsvUtils2.add_hparams(
            path_sequence,
            args.run_name,
            args.__dict__,
            metrics_mean_dict,
            epoch
        )

        if dataloader is dataloader_train:
            losses_train.append(np.mean(losses))
            R2_train.append(np.mean(R2_s))

        else:
            losses_test.append(np.mean(losses))
            R2_test.append(np.mean(R2_s))
