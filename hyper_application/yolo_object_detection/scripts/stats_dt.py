import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import chi2_contingency, pearsonr
import statsmodels.api as sm

def statistics(df):
    '''Generates the statistics revealing correlation and independence statistics for the dataframe with plots.
    
    args: 
        df: DataFrame
    returns: 
        None
    '''
    st.subheader("Statistical Analysis")
    with st.spinner("Loading basic stats..."):
        numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns

    # Pearson correlation heatmap
    st.write("### Pearson Correlation Heatmap")
    with st.spinner("Loading pearson correlation heatmap...."):
        correlation_matrix = df[numerical_cols].corr()
        fig, ax = plt.subplots(figsize=(20, 10))
        sns.heatmap(correlation_matrix, cmap='coolwarm', fmt=".2f", ax=ax)
        st.pyplot(fig)

    # Chi-square test for independence
    st.write("### Chi-Square Test for Independence")

    with st.spinner("Chi-Square test is loading...."):
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 1:
            chi2, p, dof, expected = chi2_contingency(pd.crosstab(df[categorical_cols[0]], df[categorical_cols[1]]))
            st.write(f"Chi-square statistic: {chi2}, p-value: {p}, Degrees of freedom: {dof}")

            # Chi-square heatmap
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(pd.DataFrame(expected, index=pd.unique(df[categorical_cols[0]]), 
                                    columns=pd.unique(df[categorical_cols[1]])), fmt=".2f", cmap="YlGnBu", ax=ax)
            ax.set_title("Expected Frequencies Heatmap")
            st.pyplot(fig)

def remove_high_correlation(df, threshold=0.9):
    ''' Removes highly co-related features based on the threshold set and returns dataframe and correlated features

    args: dataframe , threshold defalut=0.9
    returns: Dataframe , feature list
    '''
    correlation_matrix = df.select_dtypes(include='number').corr()
    high_corr_features = set()
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > threshold:
                colname = correlation_matrix.columns[i]
                high_corr_features.add(colname)

    df_reduced = df.drop(columns=high_corr_features, axis=1)
    return df_reduced, high_corr_features
