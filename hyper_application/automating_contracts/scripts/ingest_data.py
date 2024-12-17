import json
import pandas as pd

def load_data(data_path,req_count=20):
    '''Returns dataframe with data converted from json path

    args: datapath , count (number of samples for dataset)
    return: dataframe
    '''
    data = json.load(open(data_path,'r'))
    text_list, hypothesis_list, labels_list , target_list = [], [], [] , []
    counter = 0

    while counter != req_count:
        for k, v in data['documents'][counter].items():
            if k == 'annotation_sets':
                for annot_value in v:
                    for ndx_value in annot_value['annotations']:
                        hypothesis_list.append(ndx_value)
                        target_list.append(annot_value['annotations'][ndx_value]['choice'])
                        text_list.append(data['documents'][counter]['text'])
                counter += 1


    if len(text_list) == len(hypothesis_list) == len(target_list):
        data_df = pd.DataFrame()
        data_df['Text'] = text_list
        data_df['hypothesis'] = hypothesis_list
        data_df['target'] = target_list

    return data_df