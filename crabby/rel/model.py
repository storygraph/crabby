from typing import List
import torch
import torch.utils.data as torch_data


class RelexModel(torch.nn.Module):
    _m: int
    _r: int
    
    # From the paper:
    # * d is the num dims of the word embeddings
    # * m is the num dims of the hidden states of the RNN
    # * r is the num dims for the relation classification layer
    def __init__(self, d: int, m: int, r: int):
        super(RelexModel, self).__init__()
        
        self._m = m
        self._r = r
        
        self.rnn = torch.nn.GRU(input_size=d, hidden_size=m, bidirectional=True)
        self.linear = torch.nn.Linear(in_features=m, out_features=r)

    def forward(self, x):
        out, _ = self.rnn(torch.squeeze(x))
        
        # they are concatenated.
        out = out[:, :self._m] + out[:, self._m:]

        out = torch.transpose(out, dim0=0, dim1=1)
        out = torch.nn.functional.max_pool1d(out, kernel_size=out.size(dim=1))
        out = torch.transpose(out, dim0=0, dim1=1)

        out = self.linear(out)
        out = torch.softmax(out, dim=1)
        
        return out


class RelexTrainer:
    _training_loader: torch_data.DataLoader
    _optimizer: torch.optim.Optimizer
    _model: RelexModel
    _epoch: int
    _minibatch_size: int

    def __init__(
        self, 
        training_loader: torch_data.DataLoader, 
        optimizer: torch.optim.Optimizer,
        model: RelexModel,
        minibatch_size: int = 64,
    ) -> None:
        self._training_loader = training_loader
        self._optimizer = optimizer
        self._model = model
        self._epoch = 0
        self._minibatch_size = minibatch_size
    
    def train_one_epoch(self) -> None:
        self._epoch +=1 
        minibatch_count = 0
        cum_loss = 0
        
        curr = 0
        
        # TODO: Improve learning speed...
        for sentence, label in self._training_loader:
            self._optimizer.zero_grad()
            
            out = self._model(sentence)
            
            if curr == 0:
                out_batch = out
                label_batch = label
            else:
                out_batch = torch.cat((out_batch, out))
                label_batch = torch.cat((label_batch, label))

            curr += 1
            
            if curr == self._minibatch_size:
                loss = torch.nn.functional.cross_entropy(out_batch, label_batch)
                loss.backward()
                
                print(loss)
                
                cum_loss += loss
                minibatch_count += 1
            
                # Adjust mode params.
                self._optimizer.step()
                
                curr = 0
            
        print(f"Average loss ---> {cum_loss / minibatch_count}")


def classify(out: torch.FloatTensor) -> int:
    max_i = 0
    max_val = 0
    
    for i, val in enumerate(out):
        if val > max_val:
            max_val = val
            max_i = i
    
    return max_i
