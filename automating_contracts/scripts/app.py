def streamlit_app():
    import streamlit as st
    from train_and_analyze import inference

    # Title and Description
    st.title("Contract Automation: Inference Module")
    st.write("Predict outcomes for contracts using AI models like Albert and DistilBERT.")

    # Input Section
    st.header("Input Contract Details")
    contract_text = st.text_area("Enter Contract Text", help="Paste or type the contract text.")
    hypothesis = st.text_area("Enter Hypothesis", help="What you want to test against the contract.")
    model = st.selectbox("Select Model", ["Albert", "DistilBERT"], help="Choose the AI model for prediction.")
    predict_button = st.button("Run Prediction")

    # Output Section
    if predict_button:
        if contract_text.strip() and hypothesis.strip():
            with st.spinner("Running inference..."):
                try:
                    predictions = inference("test_data.json", predicter_model=model.lower())
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

    # Download Results
    st.download_button(
        label="Download Predictions",
        data="Sample CSV content for now",
        file_name="predictions.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    streamlit_app()