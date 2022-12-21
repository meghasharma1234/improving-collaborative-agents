import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset

from pathlib import Path

import yaml
import pandas as pd
import datetime
import copy
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


class TrajectoryDataset(Dataset):
    """Torch dataset for trainning the RNN trajectory.
    NOTE: Expects trajectory data file to be a json file with `orient='index'` containing specified keys.
    """
    def __init__(self, mode='train', hparams={'seq_len': 3}):
        self.hparams = hparams
        path = 'data/traj_data.json'
        data = pd.read_json(path, orient='index')

        sz = len(data.index)
        if mode == 'train':
            data = data[:int(.7 * sz)]
        elif mode == 'test':
            data = data[int(.7 * sz) : int(.85 * sz)]
        elif mode == 'val':
            data = data[int(.85 * sz)]
        
        self.trajTstep_data = np.array(data['trajTstep'].tolist())      # time step in the current trajectory 
        self.state_data = np.array(data['state'].tolist())
        self.action_data = np.array(data['action'].tolist())

        assert(len(self.trajTstep_data) == len(self.state_data) == len(self.action_data))
    
    def __len__(self):
        return self.action_data.shape[0]
    
    def __getitem__(self, index):
        """Retrieves a tuple for the history of states at time [t-l ... t] and action at time t

        Args:
            index (int): index of the action in the dataset

        Returns:
            states -> [seq_len, state_dim]
            action -> [action_dim]
        """

        if self.trajTstep_data[index] >= self.hparams['seq_len']:
            states = self.state_data[index - self.hparams['seq_len'] : index]
        else:
            states = self.state_data[index]
            states = states.repeat(self.hparams['seq_len'], 1)
        
        action = self.action_data[index]
        
        states = torch.from_numpy(states).type(torch.float32)
        action = torch.from_numpy(action).type(torch.float32)
        return states, action


class RNN1(nn.Module):
    """Agent architechture to map a sequence of states to action
    """
    def __init__(self, hparams, device='cuda'):
        super().__init__()
        self.hparams = hparams
        self.device = device
        self.example_input_array = torch.randn((32, 3, 4))

        self.rnn = nn.GRU(input_size=hparams['rnn_input_size'], hidden_size=hparams['rnn_hidden_size'], num_layers=1, batch_first=True)
        self.fc = nn.Linear(hparams['rnn_hidden_size'], hparams['action_size'])

        print('RNN params:', sum(p.numel() for p in self.rnn.parameters() if p.requires_grad))

    def forward(self, x):
        """Outputs action given the history of states

        Args:
            x (tensor): history of states -> [bt_sz, seq_len, state_dim]

        Returns:
            action (tensor) -> [bt_sz, action_dim]
            final hidden output (tensor) -> [bt_sz, hidden_dim]
        """
        batch_sz = x.shape[0]
        
        h_0 = torch.zeros(1, batch_sz, self.hparams['rnn_hidden_size']).to(self.device)
        out, h = self.rnn(x, h_0)
        action = self.fc(F.leaky_relu(out[:,-1]))
        return action, h


class TrainerTrajRNN(object):
    """Trainer class for predicting the action given the current state
    """
    def __init__(self, hparams) -> None:
        
        self.hparams = hparams
        self.directory = './logs/' + datetime.datetime.now().strftime("%m%d_%H_%M/")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        ### data loader
        self.loader_train = torch.utils.data.DataLoader(TrajectoryDataset('train', hparams), batch_size=hparams['batch_size'],
                                                        shuffle=True, num_workers=8)
        self.loader_test = torch.utils.data.DataLoader(TrajectoryDataset('test', hparams), batch_size=hparams['batch_size'],
                                                       shuffle=True, num_workers=8)
        self.loader_val = torch.utils.data.DataLoader(TrajectoryDataset('val', hparams), batch_size=hparams['batch_size'],
                                                       shuffle=True, num_workers=8)

        self.net = RNN1(self.hparams, self.device).to(self.device)

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=float(hparams['learning_rate']), weight_decay=float(hparams['weight_decay']))
        self.writer = SummaryWriter(self.directory)

        print(hparams)
        ### save misc. info about the trainning
        with open(self.directory + 'architecture_config.txt', 'w') as f:
            s = "model:\n" 
            for name, layer in self.net.named_modules():
                s += f'{name}: {str(layer)}\n'
            
            s += "**************\nHyperparamters:\n"
            for key, value in hparams.items():
                s += f'{key}: {str(value)}\n'
            f.write(s)

    def train(self):
        """Trains the NN model and evalutes it at given intervals 
        """
        self.test(0, mode='test')
        self.test(0, mode='val')
        for epoch in tqdm(range(self.hparams['num_epochs'])):
            epoch_running_loss = 0
            for i, data in enumerate(self.loader_train, 0):
                x, y = data
                x = x.to(self.device)
                y = y.to(self.device)
                self.optimizer.zero_grad()

                ### forward + backward + optimize
                outputs, h = self.net(x)
               
                ### optimization
                loss = self.criterion(outputs, y)
                loss.backward()
                self.optimizer.step()

                epoch_running_loss += loss.item()
            self.writer.add_scalar('Loss/train_epoch', epoch_running_loss / (i + 1), epoch)

            if (epoch + 1) % self.hparams['test_interval'] == 0:
                self.test(epoch, mode='test')
                self.test(epoch, mode='val')
                state = {
                    'epoch': epoch,
                    'state_dict': self.net.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'loss': epoch_running_loss / (i + 1)
                }
                torch.save(state, self.directory + f'model_TrajRNN_{epoch}.ckpt')
    
    def test(self, epoch, mode='test'):
        """Evalutes the model on the test/eval mode

        Args:
            epoch (int): current epoch number
            mode (str, optional): test or eval mode. Defaults to 'test'.
        """
        net = copy.deepcopy(self.net)
        net.eval()
        if mode == 'test':
            data_loader = self.loader_test
        elif mode == 'val':
            data_loader = self.loader_val
        
        running_loss = 0
        with torch.no_grad():
            for i, data in enumerate(data_loader, 0):
                x, y = data
                x = x.to(self.device)
                y = y.to(self.device)

                outputs, h = net(x)
                loss = self.criterion(outputs, y)

                running_loss += loss.item()
            self.writer.add_scalar(f'Loss/{mode}_epoch', running_loss / (i + 1), global_step=epoch)



if __name__ == "__main__":

    hparams = yaml.safe_load(Path('./hparams/hparams.yaml').read_text())
    print(hparams)

    model = RNN1(hparams, device='cuda').to('cuda')
    out, h = model(model.example_input_array.to('cuda'))
    print('output shape', out.shape, h.shape)

    # trainer = TrainerTrajRNN(hparams)
    # trainer.train()
