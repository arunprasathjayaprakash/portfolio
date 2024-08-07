import pandas as pd
import matplotlib.pyplot as plt
def visualize_stats(data,datatype,column,target):
    '''Displays plot for varius visualization

    args: Data , datatype , columna name , target variable
    return: None
    '''
    doc_lengths = [len(value) for value in data[column]]
    mean , median =  pd.Series(doc_lengths).mean() , pd.Series(doc_lengths).median()
    min_value , max_value = pd.Series(doc_lengths).min() , pd.Series(doc_lengths).max()
    plt.hist(data[column])
    plt.xlabel('Feature Length')
    plt.ylabel('Count')
    plt.show()
def plot_distribution(data,labels,datatype):
    '''Displays the distribution of the column data given

    args: data , labels (column name)
    returns: None
    '''
    plt.bar(data[labels])
    plt.show()

    