import torch
from transformers import AutoTokenizer , AutoModelForSequenceClassification

class ContractNLIDataset(torch.utils.data.Dataset):
    def __init__(self,encodings,labels=[]):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(value[idx]) for key, value in self.encodings.items()}
        # Here, ensure the label is an integer tensor
        item['labels'] = torch.tensor(int(self.labels[idx]))
        return item
    def __len__(self):
        return len(self.labels)


def tokenize_selector(tokenzier="albert"):
    ''' Returns loaded model tokenizer and model classifier that is pretrained

    args: tokenizer (default albert)
    returns: Object with loaded model
    '''

    #we use ALBERT(AL-BERT) MLM model and DistilBERT
    if tokenzier == "albert":
        model_tokenizer = AutoTokenizer.from_pretrained('albert-base-v2')
        model_classifier = AutoModelForSequenceClassification.from_pretrained(
            'albert-base-v2',num_labels = 3)
        return model_tokenizer , model_classifier
    else:
        dis_model_tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        dis_model_classifier = AutoModelForSequenceClassification.from_pretrained(
            'distilbert-base-uncased',num_labels = 3)
        return dis_model_tokenizer , dis_model_classifier

def tokenizer_train(text, hypothesis_data, max_length=256, truncation=True, model_tokenizer='albert'):
    '''Returns encoded input based on tokenizer selection

    args: text , hypothesis of contract, maximmum length , turncation , model name
    returns: json
    '''

    if model_tokenizer == 'albert':
        tokenizer,classifier = tokenize_selector()
    elif model_tokenizer == 'distilbert':
        tokenizer,classifier = tokenize_selector(tokenzier='distilbert')

    encoded_input = tokenizer(text=text,text_pair=hypothesis_data,padding='max_length',max_length=max_length,
                              truncation=truncation)
    return encoded_input , classifier

def tokenize_predict(model_path):

    classifier_pretrained = AutoModelForSequenceClassification.from_pretrained(model_path,
                                                                           local_files_only=True)
    tokenizer_pretrained = AutoTokenizer.from_pretrained(model_path,
                                              local_files_only=True)
    return tokenizer_pretrained , classifier_pretrained

