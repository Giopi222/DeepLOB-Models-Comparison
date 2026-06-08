# DeepLOB Models Comparison

This project compares two deep learning architectures for mid-price movement prediction using Limit Order Book (LOB) data:

1. **DeepLOB CNN-LSTM** (Zhang et al, 2018);
2. **CNN-Transformer Encoder**, a modified architecture in which the LSTM component is replaced by a Transformer encoder.

The motivation behind the CNN-Transformer model is to investigate whether self-attention mechanisms can match or improve the temporal modeling capabilities of LSTMs within the DeepLOB framework. Unlike recurrent architectures, Transformer encoders can model long-range dependencies by allowing each token to attend to all other tokens in the sequence. Indeed, they can be seen as an "evolution" of the LSTMs.

## Project Overview

Predicting the future movement of the mid-price is a central task in market microstructure modeling and high-frequency trading.

The mid-price is defined as: $$m_t = \frac{p^{ask}_t + p^{bid}_t}{2} $$

where $p^{ask}_t$ and $p^{bid}_t$ are respectively the best ask and best bid prices at time $t$.

This project uses supervised learning to classify the future state of the mid-price into one of three classes:

- `0`: stationary mid-price;
- `1`: downward mid-price movement;
- `2`: upward mid-price movement.

The target variable is provided over five prediction horizons (10, 20, 30, 50, 100).


## Dataset

The dataset used in this project is **FI-2010**, a publicly available benchmark dataset for Limit Order Book modeling.

The input consists of LOB snapshots represented as a sequence of historical observations. Each sample is structured as $T \times F$, where $T$ is the lookback window length, and $F$ is the number of LOB features.

In the standard DeepLOB setting, the input shape is usually $100 \times 40$, corresponding to 100 historical events and 40 order book features [($P_{ask}$, $V_{ask}$, $P_{bid}$, $V_{bid}$) $\times 10$].


    dataset: FI2010 (NoAuction --> ZScore)
    data source: https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649/data

    - train set: CF_7
    - test set:  CF_7, CF_8, CF_9



## Experimental Results

The experiments show that the original DeepLOB CNN-LSTM architecture still achieves better predictive performance and shorter training times.

The CNN-Transformer model, however, appears to converge faster during training. This suggests that self-attention may learn useful temporal representations efficiently, although in this setup it does not outperform the LSTM-based architecture overall.



**CNN-LSTM**:
- Test loss: 0.578
- Test accuracy: 0.776

              precision    recall  f1-score  

           0     0.7606    0.6846    0.7206    
           1     0.8242    0.8563    0.8399     
           2     0.6992    0.7243    0.7115     

**CNN-T**:
- Test loss: 0.626
- Test accuracy: 0.748

              precision    recall  f1-score  

           0     0.6901    0.6680    0.6789    
           1     0.8123    0.8413    0.8265     
           2     0.6813    0.6594    0.6702     



In summary:

| Model | Predictive Performance | Training Time | Convergence |
|---|---:|---:|---:|
| DeepLOB CNN-LSTM | Slightly better | Faster | Slower |
| CNN-Transformer Encoder | Slightly worse | Slower | Faster |

These results suggest that replacing the LSTM with a Transformer Encoder is a promising direction, but it may require further tuning to consistently outperform the original DeepLOB model.



## Repository Structure

```text
.
├── models/               # Model architectures (and saved models)
├── data_loader.py        # data preparation 
├── eda.py                # quick exploratory data analysis
├── train_models.py       # Training
├── test_models.py        # Test and Results
├── README.md             # Project documentation
