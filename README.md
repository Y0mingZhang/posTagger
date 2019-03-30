# tagGED!

A GloVe + BiGRU part-of-speech tagger.

## Getting Started
```
git clone https://github.com/Y0mingZhang/tagGED.git
```
Download GloVe vector from http://nlp.stanford.edu/data/glove.6B.zip and leave glove.6B.300d.txt in your working directory.

### Prerequisites

Pytorch & NLTK

### Retraining the model
```
python train.py
```
Users can modify hyperparameters at config.py.

## Usage(of predict.py)
```python
>>> from predict import Predictor
>>> p = Predictor()
>>> p.predict('In corpus linguistics, part-of-speech tagging (POS tagging or PoS tagging or POST), also called grammatical tagging orword-category disambiguation, is the process of marking up a word in a text (corpus) as corresponding to a particular part of speech,based on both its definition and its context—i.e., its relationship with adjacent and related words in a phrase, sentence, or paragraph.')
In IN corpus NNP linguistics NNP , , part-of-speech NNP tagging NNP (POS NNP tagging NNP or CC PoS NNP tagging NNP or CC POST) NNP , , also RB called VBN grammatical NNP tagging NNP or CC word-category NNP disambiguation NNP , , is VBZ the DT process NN of IN markingNN up RP a DT word NN in IN a DT text NN (corpus NN ) NN as IN corresponding NN to TO a DT particular JJ part NN of IN speech NN , , based VBN on IN both DT its PRP$ definition NN and CC its PRP$ context—i.e. NN , , its PRP$ relationship NN with IN adjacent NN and CCrelated JJ words NNS in IN a DT phrase NN , , sentence NN , , or CC paragraph NN . .
>>> p.tag_lookup('PRP$')
TAG: PRP$
Definition: Possessive pronoun

>>> p.tag_lookup('RP')
TAG: RP
Definition: Particle

>>> p.predict('GloVe is an unsupervised learning algorithm for obtaining vector representations for words.')
GloVe NNP is VBZ an DT unsupervised NN learning NNP algorithm NNP for IN obtaining VBG vector NN representations NN for IN words NNS . .
>>> p.tag_lookup('VBG')
TAG: VBG
Definition: Verb, gerund or present participle
```
