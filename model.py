import os
import torch
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import time
import copy

def train_model(device, model, dataloaders, dataset_sizes, criterion, optimizer, num_epochs=25):
    '''train_model credit to pytorch tutorial''' 
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['training', 'validation']:
            if phase == 'training':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            running_count = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'training'):

                    
                    
                    outputs = model(inputs)
                    
                    max_seq_length = outputs.size()[1]


                    labels = labels[:,:max_seq_length].contiguous().view(-1)
                    preds = torch.argmax(outputs, dim = 2)
                    preds = preds.view(-1)

                    outputs = outputs.view(-1, outputs.size()[2])


                    
                    
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'training':
                        loss.backward()
                        optimizer.step()

                # statistics
                padCount = torch.sum(labels.data == -1)
                
                preds[labels.data == -1] = -1
                
                

                running_count += labels.size()[0] - padCount
                running_loss += loss.item() * (labels.size()[0] - padCount)
                running_corrects += torch.sum(preds == labels.data) - padCount
                

            epoch_loss = running_loss.double() / running_count.double()
            epoch_acc = running_corrects.double() / running_count.double()

            print('{} Loss: {} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'validation' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())



    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model



def test(device, model, testloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            data, labels = data
            data = data.to(device)
            labels = labels.to(device)
            outputs = model(data)

            outputs=outputs.view(output.size()[0]*outputs.size()[1],-1)
            labels = labels.view(output.size()[0],-1)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the test data: %d %%' % (
        100 * correct / total))

class BiGRU(nn.Module):
    ''' Architecture for Pool NN '''
    def __init__(self, pretrained_embedding, gru_hidden_dim, gru_num_layers, num_tags):
        super(BiGRU, self).__init__()

        gru_input_dim = pretrained_embedding.size()[1]

        self.embedding = nn.Embedding.from_pretrained(pretrained_embedding)

        self.embedding.weight.requires_grad=True

        self.gru = torch.nn.GRU(gru_input_dim, gru_hidden_dim, gru_num_layers, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(gru_hidden_dim * 2, num_tags)

        self.softmax = nn.Softmax(dim = 2)
    def forward(self, x):
        length_of_seq = []

        prevZeroPos = len(x[0]) - 1
        for i in range(0, len(x)):
            prevZeroPos -= 1
            while x[i][prevZeroPos] == -1:
                prevZeroPos -= 1
            prevZeroPos += 1
            length_of_seq.append(prevZeroPos+1)


        x[x==-1] = 0
        x = self.embedding(x)

        x = torch.nn.utils.rnn.pack_padded_sequence(x, torch.LongTensor(length_of_seq), batch_first = True)


        x, h_n = self.gru(x)



        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True, padding_value=-1)


        # x size should be batch * seq *  2 * features
        
        x = x.contiguous()

        batch_size = x.size()[0]
        seq_length = x.size()[1]
        x = x.view(batch_size * seq_length, -1)

        x = self.linear(x)

        x = x.view(batch_size, seq_length, -1)
        return self.softmax(x)

