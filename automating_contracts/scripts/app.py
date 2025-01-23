def streamlit_app():
    """Runs and hosts automating contracts application

    args: None
    return: None
    """
    import streamlit as st
    from train_and_analyze import infer_data
    import json

    st.title("Contract Automation Using BERT ")
    st.write("Predict outcomes for contracts using AI models like Albert and DistilBERT. (Transformer models from Hugging face)")

    st.info("""
        ### About This Project
        This application helps you analyze  legal contracts using advanced AI models like **Albert** and **DistilBERT**. 
        With just a few clicks, you can test hypotheses about your contract text and get AI-powered predictions saving human time and efforts.

        **Key Features**:
        - **Contract Analysis**: Easily evaluate contract clauses and predict outcomes using AI.
        - **AI Models**:
          - **Albert**: Efficient and optimized for understanding complex language.
          - **DistilBERT**: A faster and lightweight model offering excellent performance.

        **How It Works**:
        1. Input your contract text and a hypothesis (e.g., "Does this contract include a confidentiality clause?").
        2. Select the AI model to use.
        3. Run the prediction to see results instantly!

        **Who Is It For?**
        Legal professionals, contract managers, and businesses looking to automate contract reviews.
            

            ** Run Submit to see the models predictions for the sample information**    
        """)

    import os

    with open(os.path.join(os.getcwd(),'scripts/sample_values.json'),'r') as file:
        example_value = json.load(file)
    
    st.header("Input Contract Details")
    contract_text = st.text_area("Enter Contract Text", help="Paste or type the contract text.",
                                 value=example_value['value'])
    hypothesis = st.text_area("Enter Hypothesis", help="What you want to test against the contract.",
                              value=example_value['hypothesis'])
    model = st.selectbox("Select Model", ["Albert", "DistilBERT"], help="Choose the AI model for prediction.")
    predict_button = st.button("Run Prediction")

 
    if predict_button:
        if contract_text.strip() and hypothesis.strip():
            with st.spinner("Running inference..."):
                try:
                    if model == 'Albert':
                        model = "albert_model"
                    else:
                        model = "distill_model"

                    predictions = infer_data(contract_text,hypothesis, predicter_model=model.lower())
                    st.success("Prediction Completed!")
                    st.write("**Prediction Results**")
                    st.json(predictions)
                except Exception as e:
                    st.error(f"Error during inference: {e}")
        else:
            st.warning("Please enter both contract text and hypothesis.")

    # Advanced Options
    with st.expander("Advanced Options"):
        st.write("Here you can inspect technical details of the model's predictions.")
        # Placeholder for tokenized inputs or attention maps (optional)
        st.write("Tokenized Inputs: Coming Soon!")

if __name__ == "__main__":
    streamlit_app()