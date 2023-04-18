# NLP Final Project

import numpy as np 
import pandas as pd
from sklearn.metrics import accuracy_score

from transformers import BertTokenizer, BertModel

import torch
from torch import cuda
from tqdm import tqdm_notebook as tqdm
device = 'cuda' if cuda.is_available() else 'cpu'

# Split data into train and test
from sklearn.model_selection import train_test_split

df = pd.read_csv('suicide_vs_depression_vs_generic.csv')

train, test = train_test_split(df, test_size=0.2, random_state=0)
train.reset_index(drop=True, inplace=True)
test.reset_index(drop=True, inplace=True)

train_X = train['selftext'].values
train_y = train['label'].values
test_X = test['selftext'].values
test_y = test['label'].values

np.shape(train_X), np.shape(train_y), np.shape(test_X), np.shape(test_y)

# not needed for training or evaluation, but useful for mapping examples
labels = {
    0: 'Suicidal',
    1: 'Depressed',
    2: 'Other'
}

# Fine-tune BERT on the dataset

#   Torch Datasets
# 
# - takes in inputs and outputs/labels
# - interfaces with tokenizer
# - handles batching


class MultiLabelDataset(torch.utils.data.Dataset):

    def __init__(self, text, labels, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.text = text
        self.targets = labels
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = self.text[index]
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            truncation=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }


#   Bert Class
# 
# - first "layer" is a pre-trained BERT model
# - you can add whatever layers you want after that


class BERTClass(torch.nn.Module):
    def __init__(self, NUM_OUT):
        super(BERTClass, self).__init__()
                   
        self.l1 = BertModel.from_pretrained("bert-base-uncased") # roberta-base # bert-base-uncased
#         self.pre_classifier = torch.nn.Linear(768, 256)
        self.classifier = torch.nn.Linear(768, NUM_OUT)
#         self.dropout = torch.nn.Dropout(0.5)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
#         pooler = self.pre_classifier(pooler)
#         pooler = torch.nn.Tanh()(pooler)
#         pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        output = self.softmax(output)
        return output


def loss_fn(outputs, targets):
    return torch.nn.CrossEntropyLoss()(outputs, targets.long())


def train(model, training_loader, optimizer):
    model.train()
    for data in tqdm(training_loader):
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
        targets = data['targets'].to(device, dtype = torch.float)

        outputs = model(ids, mask, token_type_ids)

        optimizer.zero_grad()
        loss = loss_fn(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss


def validation(model, testing_loader):
    model.eval()
    fin_targets=[]
    fin_outputs=[]
    with torch.no_grad():
        for data in tqdm(testing_loader):
            targets = data['targets']
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            outputs = model(ids, mask, token_type_ids)
            outputs = torch.sigmoid(outputs).cpu().detach()
            fin_outputs.extend(outputs)
            fin_targets.extend(targets)
    return torch.stack(fin_outputs), torch.stack(fin_targets)


#  The Tokenizer
# 
# - Converts a raw string to the ids, masks, and token_type_ids

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# what does the tokenizer do?

tokenizer.encode_plus(
            train_X[1],
            None,
            add_special_tokens=True,
            max_length=128,
            pad_to_max_length=True,
            truncation=True,
            return_token_type_ids=True
        )


# ### Training setup
# 
# - hyperparameters
# - setup dataset
# - setup parameters
# - setup dataloader


MAX_LEN = 128
BATCH_SIZE = 64
EPOCHS = 3
NUM_OUT = 3 # binary task
LEARNING_RATE = 2e-05

training_data = MultiLabelDataset(train_X, torch.from_numpy(train_y), tokenizer, MAX_LEN)
test_data = MultiLabelDataset(test_X, torch.from_numpy(test_y), tokenizer, MAX_LEN)

train_params = {'batch_size': BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

test_params = {'batch_size': BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }    

training_loader = torch.utils.data.DataLoader(training_data, **train_params)
testing_loader = torch.utils.data.DataLoader(test_data, **test_params)


# Train,  Evaluate
# 
# - model.to -> send to GPU, if available (anything computed should be put onto the GPU)
# - setup optimizer - could use Stochastic Gradient Descent, but ADAM tends to work better
# - for each epoch, train, show the loss, evaluate on the test data

load_from_saved_model = True
saved_model_path = 'F:\\GitHub\\nlp_final_project\\flask-server\\saved_model'

model = BERTClass(NUM_OUT)
model.to(device)

if load_from_saved_model:
    model = torch.load(saved_model_path)
else:
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        loss = train(model, training_loader, optimizer)
        print(f'Epoch: {epoch}, Loss:  {loss.item()}')  
        guess, targets = validation(model, testing_loader)
        guesses = torch.max(guess, dim=1)
        print('accuracy on test set {}'.format(accuracy_score(guesses.indices, targets)))
    # Save Model so that it can be loaded and not retrained
    torch.save(model, saved_model_path)

# ### Evaluation of Model

guess, targets = validation(model, testing_loader)
guesses = torch.max(guess, dim=1)
print('accuracy on test set {}'.format(accuracy_score(guesses.indices, targets)))