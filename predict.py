from preprocessing import preprocess
from model import train_model, test, BiGRU
import torch
import torch.nn as nn
import torch.optim as optim
import json
from config import MODEL_PARAMS


def save_corpus():
    glove_pretrained, dataloaders, dataset_sizes, tbl, tagset, reverse_tagset, tag_definitions = preprocess()
    params={'tbl':tbl, 'tagset':tagset, 'reverse_tagset':reverse_tagset,'tag_definition':tag_definitions}
    json_params = json.dumps(params)
    f = open('params.json', 'w', encoding='utf8')
    f.write(json_params)
    f.close()
    return params


class Predictor():
    def __init__(self):
        try:
            f = open("params.json","r", encoding='utf8')
            self.params = json.loads(f.read())
        except FileNotFoundError:
            self.params = save_corpus()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.model = BiGRU(torch.zeros((len(self.params['tbl'])+1,300)), MODEL_PARAMS['gru_hidden_dim']
        , MODEL_PARAMS['gru_num_layers'], len(self.params['tagset']), MODEL_PARAMS['concat']).to(device)
        self.model.load_state_dict(torch.load('trained_model.pt', map_location=lambda storage, loc: storage))
        self.model.eval()

    def predict(self, sentence):
        words = sentence.split()

        lis = []
        new_words = []
        for word in words:
            symbol = None
            if not word[-1].isalnum():
                symbol = word[-1]
                word = word[:-1]
            if word.lower() in self.params['tbl']:
                lis.append(self.params['tbl'][word.lower()])
            else:
                lis.append(0)
            new_words.append(word)
            if symbol != None:
                if symbol in self.params['tbl']:
                    lis.append(self.params['tbl'][symbol])
                else:
                    lis.append(0)
                new_words.append(symbol)

        x = torch.LongTensor(lis).to(self.device)
        x = x.unsqueeze(0)
        y_raw = self.model(x)
        y_pred = torch.argmax(y_raw, dim=2).view(-1)
        tagged_sent = ''
        for i in range(len(y_pred)):
            tagged_sent += new_words[i]
            tagged_sent += ' '
            tagged_sent += self.params['reverse_tagset'][y_pred[i]]
            tagged_sent += ' '
        print(tagged_sent)
    
    def tag_lookup(self, tag):
        try:
            print('TAG:',tag)
            print('Definition:', self.params['tag_definition'][tag][0])
        except:
            print('Error: Tag not found.')



if __name__ == '__main__':
    main()


