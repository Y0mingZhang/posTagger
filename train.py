from preprocessing import preprocess
from model import train_model, test, BiGRU
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

def main():


    glove_pretrained, dataloaders, dataset_sizes, tbl, tagset = preprocess()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = BiGRU(glove_pretrained, 10, 1, len(tagset)).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index = -1)
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    train_model(device, net, dataloaders, dataset_sizes, criterion, optimizer, num_epochs=1)
    test(device, net, dataloaders['testing'])




if __name__ == '__main__':
    main()  