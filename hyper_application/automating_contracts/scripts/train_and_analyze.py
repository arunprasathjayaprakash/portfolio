import numpy as np
import os

import pandas as pd
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


def label_mapper(prediction):
    if prediction == 0:
        return ("NotMentioned:"
                "A particular issue, fact, or argument is not addressed or referenced in the text, contract, law, or"
                " testimony being examined."
                "Legal Example:"
                "A contract states, The tenant shall pay rent on the 1st of every month."
                "The contract does not mention who is responsible for property maintenance."
                "In this case, the maintenance obligation is not mentioned and may require further negotiation or "
                "interpretation by the court.")

    elif prediction == 1:
        return ("Entailment"
                "A statement, clause, or provision in a legal document logically or necessarily follows from another "
                "statement"
                " or is implied by it. It indicates that if one legal fact is true, another related legal consequence "
                "must also be true."
                "Legal Example:"
                "A will states, 'All assets are to be equally distributed among my children.'"
                "Entailment: Each child is entitled to an equal share of the estate."
                "This legal interpretation follows directly from the explicit language of the will.")
    else:
        return ("Contradiction"
                "Meaning: A statement or clause directly opposes another within the same legal document, law, "
                "or testimony. This creates inconsistency that needs to be resolved."
                "Legal Example:"
                "A lease agreement states, 'The tenant is responsible for all utility payments,' but later states,"
                "'The landlord will cover the water bill.'"
                "These clauses are in contradiction, and the court or parties would need to clarify the intent to resolve the conflict.")
def train_and_predict(data_path,
                      valid_path,
                      test_path,
                      misclass_threshold,
                      retrain=False):
  '''Returns training report

  args: train data path , validation path , testing path , retrain option to train new model with data change
  returns: training report
  '''

  #For lesser train time a sample of small size is selected here to make it easy to train on local
  #Change count based on platform specifications
  train_data = load_data(data_path, 35) # Count - Samples required for each data split
  valid_data = load_data(valid_path, 10)
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
  train_encoded_distil, distill_classifier = tokenizer_train(text,
                                                             hyp_data,
                                                             model_tokenizer='distilbert')

  valid_data_encoded_distil, _ = tokenizer_train(valid_data['Text'].to_list(),
                                                 valid_data['hypothesis'].to_list(),
                                                 model_tokenizer='distilbert')

  test_data_encoded_distil, _ = tokenizer_train(test_data['Text'].to_list(),
                                                test_data['hypothesis'].to_list(),
                                                model_tokenizer='distilbert')

  train_data['target'] = train_data['target'].map({"NotMentioned": 0, "Entailment": 1, "Contradiction": 2})
  valid_data['target'] = valid_data['target'].map({"NotMentioned": 0, "Entailment": 1, "Contradiction": 2})
  test_data['target'] = test_data['target'].map({"NotMentioned": 0, "Entailment": 1, "Contradiction": 2})

  # Data for training , validation and testing
  train_dataset = ContractNLIDataset(train_encoded,
                                     train_data['target'].values)

  valid_dataset = ContractNLIDataset(valid_data_encoded,
                                     valid_data['target'].values)

  test_dataset = ContractNLIDataset(test_data_encoded,
                                    test_data['target'].values)

  # Data for distill birt
  train_dataset_distill = ContractNLIDataset(train_encoded_distil,
                                             train_data['target'].values)

  valid_dataset_distill = ContractNLIDataset(valid_data_encoded_distil,
                                             valid_data['target'].values)

  test_dataset_distill = ContractNLIDataset(test_data_encoded_distil,
                                            test_data['target'].values)

  # Training Albert Model
  train_args = TrainingArguments('models/albert_model',
                                 num_train_epochs=3,
                                 weight_decay=0.01,
                                 logging_steps=1,
                                 eval_strategy='epoch')

  train_model = Trainer(model=albert_classifier,
                        args=train_args,
                        train_dataset=train_dataset,
                        eval_dataset=valid_dataset,
                        tokenizer=AutoTokenizer.from_pretrained('albert-base-v2'),
                        compute_metrics=compute_metrics)
  train_model.train()


  # Training Distillbert Model
  train_args_distill = TrainingArguments(os.path.join(os.getcwd(), 'models/distill_model'),
                                         num_train_epochs=3,
                                         weight_decay=0.01,
                                         logging_steps=1,
                                         eval_strategy='epoch')


  train_model_distill = Trainer(model=distill_classifier,
                                args=train_args_distill,
                                train_dataset=train_dataset_distill,
                                eval_dataset=valid_dataset_distill,
                                tokenizer=AutoTokenizer.from_pretrained('distilbert-base-uncased'),
                                compute_metrics=compute_metrics)

  train_model_distill.train()

  predictions , albert_metrics = predict_and_save_results(train_model, 'allbert', test_dataset, test_data)
  predictions_distill , distill_metrics = predict_and_save_results(train_model_distill,
                                                                   'distill-bert',
                                                                   test_dataset_distill,
                                                                   test_data)

  albert_misclass = predictions[predictions[f'predicted_from_allbert'] != test_data['target'].values].iloc[:, :3]

  # For the DistilBERT model
  distilbert_misclass = predictions_distill[
                          predictions_distill[f'predicted_from_distill-bert'] != test_data['target'].values].iloc[:, :3]

  albert_misclass = (len(albert_misclass) / len(test_data['target'])) * 100
  distill_bert_misclass = (len(distilbert_misclass) / len(test_data['target'])) * 100

  #save model
  '''
  if misclass for a model is lesser than misclass_threshold save the model
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

  training_report = {"Training metrics":{'Albert':albert_metrics,
                                         "Distill_metrics":distill_metrics},

                     "Model Misclassifications":{"Albert":albert_misclass,
                                                 "Distill_bert":distill_bert_misclass}}

  return training_report

def predict_and_save_results(trainer,model_name,test_dataset,test_dataframe):
  '''Returns Test dataframe with predicted labels

  args: trainer , model name , prediction dataset , prediction dataframe object
  returns: dataframe object
  '''
  metrics , labels , predictions = trainer.predict(test_dataset)
  test_dataframe[f'predicted_from_{model_name}'] = labels
  return test_dataframe , metrics

def infer_data(text,hypothesis,predicter_model="albert_model"):

    '''Returns predictions based on selected method

    args:data path , predictor model name
    returns: predictions with the selected model
    '''
    # data = load_data(data_path,30)
    #
    # #Handling label column for internal testing
    # if 'target' in data:
    #     data.drop('target',axis=1)
    #
    # if predicter_model == 'albert':
    #   tokenizer , classifier = tokenize_predict(os.path.join(os.getcwd(),'models/albert_model'))
    #   training_args = TrainingArguments(
    #     output_dir=os.path.join(os.getcwd(), 'predictions/albert_model'),
    #     do_predict=True
    #   )
    # else:
    #   tokenizer, classifier = tokenize_predict(os.path.join(os.getcwd(), 'models/distill_model'))
    #   training_args = TrainingArguments(
    #     output_dir=os.path.join(os.getcwd(),'predictions/distill_model'),
    #     do_predict=True
    #   )
    #
    # predictor = Trainer(model=classifier,args=training_args)
    #
    # encoded = tokenizer(text=data['Text'].to_list(), text_pair=data['hypothesis'].to_list())
    # # dataset_enc = Dataset.from_dict(encoded)
    # infer_dataset = ContractNLIDataset(encoded)
    # predictions = predictor.predict(infer_dataset)
    # return predictions

    "Inference pipeline"
    # data = load_data(data_path,1)
    data = pd.DataFrame()
    data['Text'] = [text]
    data['hypothesis'] = [hypothesis]
    tokenizer, albert_classifier = tokenize_predict(os.path.join(os.getcwd(),f'models/{predicter_model}'))
    encoded = tokenizer(data['Text'].to_list(),
                        data['hypothesis'].to_list(),
                        padding='max_length',
                        truncation=True,
                        max_length=512)

    inference_data = ContractNLIDataset(encoded,
                     [0]*len(encoded.encodings))

    # inference_data.encodings["input_ids"] = inference_data.encodings["input_ids"].squeeze(0)
    # inference_data.encodings["attention_mask"] = inference_data.encodings["attention_mask"].squeeze(0)

    predict_args = TrainingArguments(output_dir=os.path.join(os.getcwd(),'predictions/albert_model'),
                                     do_predict=True)

    trainer = Trainer(model=albert_classifier,
                      args=predict_args)
    predictions = predict_and_save_results(trainer,'albert',inference_data,data)

    if predicter_model == 'albert_model':
        return {"predictions": label_mapper(predictions[0][f'predicted_from_albert'][0])}
    else:
        return {"predictions": label_mapper(predictions[0][f'predicted_from_distill'][0])}


'''
Uncomment this to test script as individual modules
'''
if __name__ == "__main__":
  data_path = r'C:\csulb_projects\portfolio_projects\hyper_application\automating_contracts\data\train.json'
  valid_path = r'C:\csulb_projects\portfolio_projects\hyper_application\automating_contracts\data\dev.json'
  test_path = r'C:\csulb_projects\portfolio_projects\hyper_application\automating_contracts\data\test.json'
  misclass_threshold = 0.2
  # train_and_predict(data_path,valid_path,test_path,misclass_threshold)
  infer_data(test_path,'','')