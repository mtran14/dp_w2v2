import os
import sys
import time
import pickle
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
# import wandb

class MinMaxScaler:

    def __init__(self, epsilon=1e-7):
        """Standard Scaler.
        The class can be used to normalize PyTorch Tensors using native functions. The module does not expect the
        tensors to be of any specific shape; as long as the features are the last dimension in the tensor, the module
        will work fine.
        :param mean: The mean of the features. The property will be set after a call to fit.
        :param std: The standard deviation of the features. The property will be set after a call to fit.
        :param epsilon: Used to avoid a Division-By-Zero exception.
        """
        self.min, self.max = None, None
        self.epsilon = epsilon

    def fit(self, values):
        dims = list(range(values.dim() - 1))
        self.min = torch.min(values)
        self.max = torch.max(values)

    def transform(self, values):
        return ((values - self.min) / (self.max - self.min + self.epsilon))

    def fit_transform(self, values):
        self.fit(values)
        return self.transform(values)

class RawGradient(Dataset):
    def __init__(self, split, path):
        files = os.listdir(path)
        y = np.ones(len(files),)
        X_train, X_test, _, _ = train_test_split(files, y, test_size=0.1, random_state=42)
        if(split=='train'):
            self.X = X_train
        else:
            self.X = X_test
        self.path = path
        self.scaler = MinMaxScaler()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        current_file = self.X[idx]
        try:
            with open(os.path.join(self.path, current_file), 'rb') as f:
                x = pickle.load(f)
        except:
            with open(os.path.join(self.path, self.X[5]), 'rb') as f:
                x = pickle.load(f)
        original_features = torch.FloatTensor(x['w2v2_features'])
        saliency_map_raw = torch.FloatTensor(x['saliency_map'])
        T, D = saliency_map_raw.size()
        saliency_map_normed = self.scaler.fit_transform(saliency_map_raw.view(T*D)).reshape(T, D)
        return original_features, saliency_map_normed

def collate_batch(batch):

  src_list, tgt_list = [], []

  for (src,_tgt) in batch:
    src_list.append(src)
    tgt_list.append(_tgt)
  return pad_sequence(src_list, batch_first=True), pad_sequence(tgt_list, batch_first=True)

class Transformer(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers):
        super(Transformer, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.model = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)

    def forward(self, x):
        return self.model(x)

def train(model, optimizer, criterion, dataloader, epoch):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.train()
    model.to(device)
    log_interval = 500
    start_time = time.time()

    for idx, (src, tgt) in enumerate(dataloader):
        optimizer.zero_grad()
        src = src.to(device)
        tgt = tgt.to(device)
        predicted_src = model(src)
        loss = criterion(predicted_src, tgt)
        #loss = ccc(predicted_src, tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.8)
        optimizer.step()
        # wandb.log({'Loss': loss.item()})
        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches '
                  '| loss {:8.3f}'.format(epoch, idx, len(dataloader),
                                              loss.item()))
            start_time = time.time()

def evaluate(model, criterion, dataloader):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    model.to(device)
    losses = []
    with torch.no_grad():
        for idx, (src, tgt) in enumerate(dataloader):
            src = src.to(device)
            tgt = tgt.to(device)
            predicted_src = model(src)
            loss = criterion(predicted_src, tgt)
            #loss = ccc(predicted_src, tgt)
            losses.append(loss.item())
    return np.mean(losses)

if __name__ == '__main__':
    saliency_maps_path = sys.argv[1]
    train_dataset = RawGradient('train', saliency_maps_path)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, collate_fn=collate_batch)
    test_dataset = RawGradient('test', saliency_maps_path)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_batch)

    model = Transformer(
        d_model=768,
        nhead=12,
        num_encoder_layers=6
    )
    # wandb.init(project="dp-speech-continuous")
    # wandb.watch(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    epochs = 60
    criterion = nn.L1Loss()
    best_loss = 10000
    for i in range(epochs):
        train(model, optimizer, criterion, train_loader, i+1)
        test_loss = evaluate(model, criterion, test_loader)
        print(f'Epoch: {i+1} - Test Loss: {test_loss}')
        if(test_loss < best_loss):
            torch.save(model.state_dict(), "masker_transformer.pt")
            best_loss = test_loss
