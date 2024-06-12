import numpy as np
import os
from sklearn.metrics import  accuracy_score , f1_score , recall_score , precision_score
from transformers import TrainingArguments
from transformers import Trainer
from transformers import AutoTokenizer , AutoModelForSequenceClassification
from ingest_data import load_data
from tokenize_data import tokenizer_train , tokenize_predict
from tokenize_data import ContractNLIDataset
from datasets import Dataset
def compute_metrics(p):
  '''Returns metrics score for predicted data

  args: predictions
  returns: prediction metrics
  '''
  preds = np.argmax(p.predictions, axis = 1)
  labels = p.label_ids

  accuracy = accuracy_score(labels, preds)
  precision = precision_score(labels, preds, average = 'macro')
  recall = recall_score(labels, preds, average = 'macro')
  f1 = f1_score(labels, preds, average = 'macro')

  return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

def train_and_predict(data_path,
                      valid_path,
                      test_path,
                      misclass_threshold,
                      retrain=False):
  '''Returns training report

  args: train data path , validation path , testing path , retrain option to train new model with data change
  returns: training report
  '''
  data_path = r'C:\csulb_projects\portfolio_projects\automating_contracts\data\train.json'
  train_data = load_data(data_path, 35)
  valid_pth = r'C:\csulb_projects\portfolio_projects\automating_contracts\data\dev.json'
  valid_data = load_data(valid_pth, 10)
  test_path = r'C:\csulb_projects\portfolio_projects\automating_contracts\data\test.json'
  test_data = load_data(test_path, 5)

  # visualize_stats(train_data,'training','Text','labels')
  text = train_data['Text'].to_list()
  hyp_data = train_data['hypothesis'].to_list()
  train_encoded, albert_classifier = tokenizer_train(text, hyp_data)
  valid_data_encoded, _ = tokenizer_train(valid_data['Text'].to_list(),
                                          valid_data['hypothesis'].to_list())
  test_data_encoded, _ = tokenizer_train(test_data['Text'].to_list(),
                                         test_data['hypothesis'].to_list())

  # DistilBert
  train_encoded_distil, distill_classifier = tokenizer_train(text, hyp_data, model_tokenizer='distilbert')
  valid_data_encoded_distil, _ = tokenizer_train(valid_data['Text'].to_list(),
                                                 valid_data['hypothesis'].to_list(), model_tokenizer='distilbert')
  test_data_encoded_distil, _ = tokenizer_train(test_data['Text'].to_list(),
                                                test_data['hypothesis'].to_list(), model_tokenizer='distilbert')

  train_data['target'] = train_data['target'].map({"NotMentioned": 0, "Entailment": 1, "Contradiction": 2})
  valid_data['target'] = valid_data['target'].map({"NotMentioned": 0, "Entailment": 1, "Contradiction": 2})
  test_data['target'] = test_data['target'].map({"NotMentioned": 0, "Entailment": 1, "Contradiction": 2})


  contract_data = ContractNLIDataset(train_encoded, train_data['target'].values)

  # testing the function for sample data
  contract_data.__getitem__(3)

  # Data for training , validation and testing
  train_dataset = ContractNLIDataset(train_encoded, train_data['target'].values)
  valid_dataset = ContractNLIDataset(valid_data_encoded, valid_data['target'].values)
  test_dataset = ContractNLIDataset(test_data_encoded, test_data['target'].values)

  # Data for distill birt
  train_dataset_distill = ContractNLIDataset(train_encoded_distil, train_data['target'].values)
  valid_dataset_distill = ContractNLIDataset(valid_data_encoded_distil, valid_data['target'].values)
  test_dataset_distill = ContractNLIDataset(test_data_encoded_distil, test_data['target'].values)

  # Training Albert Model
  train_args = TrainingArguments('models/albert_model', num_train_epochs=3, weight_decay=0.01,
                                 logging_steps=1, eval_strategy='epoch')
  train_model = Trainer(model=albert_classifier, args=train_args, train_dataset=train_dataset,
                        eval_dataset=valid_dataset, tokenizer=AutoTokenizer.from_pretrained('albert-base-v2'), compute_metrics=compute_metrics)
  train_model.train()


  # Training Distillbert Model
  train_args_distill = TrainingArguments(os.path.join(os.getcwd(), 'models/distill_model'), num_train_epochs=3,
                                         weight_decay=0.01,
                                         logging_steps=1, eval_strategy='epoch')
  train_model_distill = Trainer(model=distill_classifier, args=train_args_distill, train_dataset=train_dataset_distill,
                                eval_dataset=valid_dataset_distill, tokenizer=AutoTokenizer.from_pretrained('distilbert-base-uncased'),
                                compute_metrics=compute_metrics)
  train_model_distill.train()

  predictions , albert_metrics = predict_and_save_results(train_model, 'allbert', test_dataset, test_data)
  predictions_distill , distill_metrics = predict_and_save_results(train_model_distill, 'distill-bert', test_dataset_distill,
                                                 test_data)

  albert_misclass = predictions[predictions[f'predicted_from_allbert'] != test_data['target'].values].iloc[:, :3]

  # For the DistilBERT model
  distilbert_misclass = predictions_distill[
                          predictions_distill[f'predicted_from_distill-bert'] != test_data['target'].values].iloc[:, :3]

  albert_misclass = (len(albert_misclass) / len(test_data['target'])) * 100
  distill_bert_misclass = (len(distilbert_misclass) / len(test_data['target'])) * 100

  #save model
  '''
  if misclass for a model is near to zero save the model
  '''
  if albert_misclass <= misclass_threshold and distill_bert_misclass <= misclass_threshold:
    train_model.save_model('models/albert_model')
    train_model.save_model('models/distill_model')

  '''
  Uncomment this if you need to push the model to hugging face
  Date: 6/11/24
  Author: Arun prasath jayaprakash
  '''
  # from huggingface_hub import notebook_login
  # notebook_login()
  # train_model.push_to_hub()

  training_report = {"Training metrics":{'Albert':albert_metrics,"Distill_metrics":distill_metrics},
                     "Model Misclassifications":{"Albert":albert_misclass,"Distill_bert":distill_bert_misclass}}

  return training_report

def predict_and_save_results(trainer,model_name,test_dataset,test_dataframe):
  '''Returns Test dataframe with predicted labels

  args: trainer , model name , prediction dataset , prediction dataframe object
  returns: dataframe object
  '''
  metrics , labels , predictions = trainer.predict(test_dataset)
  test_dataframe[f'predicted_from_{model_name}'] = labels
  return test_dataframe , metrics

def inference(data_path,predicter_model="albert"):
  '''Returns predictions based on selected method

  args:data path , predictor model name
  returns: predictions with the selected model
  '''
  data = load_data(data_path,10)
  data.drop('target',axis=1)
  if predicter_model == 'albert':
    tokenizer , classifier = tokenize_predict(os.path.join(os.getcwd(),'models/albert_model'))
    training_args = TrainingArguments(
      output_dir=os.path.join(os.getcwd(), 'predictions/albert_model'),
      do_predict=True
    )
  else:
    tokenizer, classifier = tokenize_predict(os.path.join(os.getcwd(), 'models/distill_model'))
    training_args = TrainingArguments(
      output_dir=os.path.join(os.getcwd(),'predictions/distill_model'),
      do_predict=True
    )

  predictor = Trainer(model=classifier,args=training_args)

  encoded = tokenizer(text=data['Text'].to_list(), text_pair=data['hypothesis'].to_list(),truncation=True)
  dataset_enc = Dataset.from_dict(encoded)
  predictions = predictor.predict(dataset_enc)
  return predictions

'''
Uncomment this to test script as individual modules
'''
if __name__ == "__main__":
  data_path = r'C:\csulb_projects\portfolio_projects\automating_contracts\data\train.json'
  valid_path = r'C:\csulb_projects\portfolio_projects\automating_contracts\data\dev.json'
  test_path = r'C:\csulb_projects\portfolio_projects\automating_contracts\data\test.json'
  misclass_threshold = 0.2
  # train_and_predict(data_path,valid_path,test_path,misclass_threshold)
  inference(data_path)