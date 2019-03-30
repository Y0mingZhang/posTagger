from preprocessing import preprocess
from model import train_model, test, BiGRU
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from config import MODEL_PARAMS

def train():


    glove_pretrained, dataloaders, dataset_sizes, tbl, tagset, reverse_tagset, tag_definitions = preprocess()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = BiGRU(glove_pretrained, MODEL_PARAMS['gru_hidden_dim']
    	, MODEL_PARAMS['gru_num_layers'], len(tagset), MODEL_PARAMS['concat']).to(device)

    criterion = nn.NLLLoss(ignore_index = -1)
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    train_model(device, net, dataloaders, dataset_sizes, criterion, optimizer, MODEL_PARAMS['num_epochs'])
    test(device, net, dataloaders['testing'])
    torch.save(net.state_dict(), 'trained_model.pt')




if __name__ == '__main__':
    train()
