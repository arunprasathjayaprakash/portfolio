
# Automating Contracts

## Project Description

Automating contract review involves analyzing legal documents to ensure the accuracy and clarity of legal standards. Natural language inference (NLI) techniques utilize artificial intelligence to analyze the relationships between different pieces of text. The ContractNLI dataset provides a collection of contracts and corresponding hypotheses. The models are trained on this dataset, with the goal of determining whether each hypothesis is entailed by, contradicts, or is not mentioned in the contract.

## Tranformer Models Used

### ALBERT (A Lite BERT)

**Overview:**
ALBERT is a lightweight version of BERT (Bidirectional Encoder Representations from Transformers) designed to reduce the model size while maintaining performance. It was introduced by Google Research in a paper titled "ALBERT: A Lite BERT for Self-supervised Learning of Language Representations."

**Key Features:**
- **Parameter Reduction:** ALBERT reduces the number of parameters by sharing parameters across layers and using factorized embedding parameterization.
- **Sentence Order Prediction (SOP):** ALBERT introduces a new pretraining task called SOP, which helps the model understand sentence coherence.
- **Efficient Training:** By reducing the number of parameters, ALBERT achieves faster training times and lower memory consumption compared to the original BERT model.

**Applications:**
- ALBERT is used in various NLP tasks such as text classification, question answering, and natural language inference, offering a good balance between efficiency and accuracy.

### DistilBERT

**Overview:**
DistilBERT is a smaller, faster, and cheaper version of BERT, developed by Hugging Face. It aims to provide a more efficient model that retains 97% of BERT's performance while being 60% faster and using 40% fewer parameters.

**Key Features:**
- **Distillation Process:** DistilBERT is trained using a technique called knowledge distillation, where a smaller student model (DistilBERT) learns to mimic the behavior of a larger teacher model (BERT).
- **Reduced Size:** By removing certain layers and reducing the number of parameters, DistilBERT achieves significant efficiency gains without a substantial loss in performance.
- **Versatility:** Despite its reduced size, DistilBERT maintains strong performance across various NLP tasks, making it a popular choice for real-time applications.

**Applications:**
- DistilBERT is widely used in applications where computational resources are limited or when faster inference times are required, such as chatbots, mobile applications, and real-time text processing.


In this project, you’ll gain hands-on experience in natural language processing (NLP) by:

- Loading and exploring the dataset as JSON files.
- Performing Natural Language Inference (NLI) using transformer models.
- Analyzing the errors.

## Project Structure

```
automating_contracts/
│
├── data/
├── refered_publications/
├── scripts/
│   │
│   ├── models/
│   │   ├── albert_model/
│   │   └── distill_model/
│   ├── predictions/
│   ├── ingest_data.py
│   ├── tokenize_data.py
│   ├── train_and_analyze.py
│   └── visualize_features.py
│
├── requirements.txt
└── README.md
```

## Setup Instructions

1. **Clone the repository**
    ```bash
    git clone https://github.com/arunprasathjayaprakash/portfolio/tree/7576a11f3be23fc5754de996733d63b682264cbe/automating_contracts
    cd automating_contracts
    ```

2. **Create a virtual environment and activate it**
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

3. **Install the dependencies**
    ```bash
    pip install -r requirements.txt
    ```

## Scripts Overview

### Ingest Data

- **File**: `ingest_data.py`
- **Description**: This script loads and explores the dataset from JSON files.

### Tokenize Data

- **File**: `tokenize_data.py`
- **Description**: This script tokenizes the data using the tokenizer associated with the chosen transformer model.

### Train and Analyze

- **File**: `train_and_analyze.py`
- **Description**: This script trains the NLI model and performs error analysis on the results.

### Visualize Features (optional)

- **File**: `visualize_features.py`
- **Description**: This script visualizes the features and errors in the model predictions.

## Models

- **Directory**: `scripts/models`
- **Description**: Contains directories for different transformer models (`albert_model`, `distill_bert`, and `distill_model`).

## Predictions

- **Directory**: `scripts/predictions`
- **Description**: Contains the predictions for the models.

## Data

- **Directory**: `Data`
- **Description**: Data from ContractNLI website was used.

## Referred publications

- **Directory**: `refered_publications`
- **Description**: Contains the referred publications in PDF format.

## Running the Project

1. **Ingest Data**:
    ```bash
    python scripts/ingest_data.py
    ```

2. **Tokenize Data**:
    ```bash
    python scripts/tokenize_data.py
    ```

3. **Train and Analyze**:
    ```bash
    python scripts/train_and_analyze.py
    ```

4. **Visualize Features**:
    ```bash
    python scripts/visualize_features.py
    ```

## Contributing

Contributions are welcome! Please submit a pull request or open an issue to discuss any changes.
