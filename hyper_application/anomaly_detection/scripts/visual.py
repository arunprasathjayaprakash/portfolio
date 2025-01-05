import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns

def plot_data_distribution(df):
    ''' Returns data distribution for the dataframe

    args: Dataframe
    returns: None , displays various distribution
    
    '''
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
    df[numerical_columns].hist(bins=30, figsize=(20, 15), color='skyblue', edgecolor='black')
    plt.suptitle('Distribution of Numerical Features', fontsize=16)
    st.pyplot(plt)


def cat_distribution(df):
    ''' Plots categorical distribution for the dataframe

    args: dataframe
    returns: None
    
    '''
    categorical_columns = ['protocol_type', 'service', 'flag', 'label']
    for col in categorical_columns:
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df, x=col, order=df[col].value_counts().index, palette='viridis')
        plt.title(f'Count Plot for {col}', fontsize=16)
        plt.xlabel(col, fontsize=14)
        plt.ylabel('Count', fontsize=14)
        plt.xticks(rotation=90, fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        st.pyplot(plt)


def plot_confusion_matrix(cm, target_names, title='Confusion Matrix', cmap=plt.cm.Greens):
    ''' Plots confusion matrix
    
    '''
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=90)
    plt.yticks(tick_marks, target_names)
    plt.tight_layout()

    width, height = cm.shape
    for x in range(width):
        for y in range(height):
            plt.annotate(str(cm[x][y]), xy=(y, x), 
                        horizontalalignment='center',
                        verticalalignment='center')
            
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')