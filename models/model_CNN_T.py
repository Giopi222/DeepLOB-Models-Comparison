
import tensorflow as tf

from tensorflow.keras.models import Model  
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    LeakyReLU,
    MaxPooling2D,
    concatenate,
    Reshape,
    Dense,
    Dropout,
    LayerNormalization,
    MultiHeadAttention)


'''
CNN-Transformer 

Input 100 x 40                         Input(shape=(T, F, 1))

Convolutional feature extractor
--------------------------------
1x2@N stride 1x2                       Conv2D(N, (1,2), strides=(1,2))
4x1@N                                  Conv2D(N, (4,1), padding="same")
4x1@N                                  Conv2D(N, (4,1), padding="same")

1x2@N stride 1x2                       Conv2D(N, (1,2), strides=(1,2))
4x1@N                                  Conv2D(N, (4,1), padding="same")
4x1@N                                  Conv2D(N, (4,1), padding="same")

1x10@N                                 Conv2D(N, (1,10))
4x1@N                                  Conv2D(N, (4,1), padding="same")
4x1@N                                  Conv2D(N, (4,1), padding="same")


Inception module
----------------
Branch 1: 1x1@M -> 3x1@M
Branch 2: 1x1@M -> 5x1@M
Branch 3: 3x1 maxpool -> 1x1@M

Concatenate branches                   output shape: (T, 1, 3M)
Reshape                                output shape: (T, 3M)


Transformer temporal encoder
----------------------------
Learnable positional embedding          PositionalEmbedding(T, 3M)

Transformer Encoder block x L:
    Multi-Head Self-Attention
    Dropout
    Residual connection + LayerNorm
    Feed-Forward Network:
        Dense(ff_dim)
        LeakyReLU
        Dropout
        Dense(3M)
        Dropout
    Residual connection + LayerNorm


Temporal aggregation
--------------------
GlobalAveragePooling1D                  aggregates information over T timesteps


Classification head
-------------------
Dense(dense_num)
LeakyReLU
Dropout
Dense softmax                           Dense(3, activation="softmax")
'''


class PositionalEmbedding(tf.keras.layers.Layer):
    """
    Learnable positional embedding.

    Adds a trainable vector to each timestep so that the Transformer
    can distinguish early, middle and late events in the lookback window.
    """

    def __init__(self, sequence_length, embedding_dim):
        super().__init__()
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim

        self.position_embedding = tf.keras.layers.Embedding(
            input_dim=sequence_length,
            output_dim=embedding_dim
        )

    def call(self, inputs):
        sequence_length = tf.shape(inputs)[1]

        positions = tf.range(
            start=0,
            limit=sequence_length,
            delta=1
        )

        embedded_positions = self.position_embedding(positions)

        return inputs + embedded_positions


def transformer_encoder_block(
    inputs,
    num_heads,
    key_dim,
    ff_dim,
    dropout_rate=0.1,
    leaky_relu_alpha=0.01
):
    attention_output = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=key_dim,
        dropout=dropout_rate
    )(
        query=inputs,
        value=inputs,
        key=inputs
    )

    attention_output = Dropout(dropout_rate)(attention_output)

    x = LayerNormalization(epsilon=1e-6)(inputs + attention_output)

    ff_output = Dense(ff_dim)(x)
    ff_output = LeakyReLU(negative_slope=leaky_relu_alpha)(ff_output)
    ff_output = Dropout(dropout_rate)(ff_output)
    ff_output = Dense(inputs.shape[-1])(ff_output)
    ff_output = Dropout(dropout_rate)(ff_output)

    x = LayerNormalization(epsilon=1e-6)(x + ff_output)

    return x


def initiate_CNN_Transformer_model(
    lookback_timestep,
    feature_num,
    conv_filter_num,
    inception_num,
    transformer_num_heads,
    transformer_key_dim,
    transformer_ff_dim,
    transformer_num_layers,
    dense_num,
    leaky_relu_alpha,
    dropout_rate,
    loss,
    optimizer,
    metrics
):
    
    input_tensor = Input(shape=(lookback_timestep, feature_num, 1)) # x \in R^{TxFx1}


    # Conv block 1
    conv_layer1 = Conv2D(conv_filter_num, (1, 2), strides=(1, 2))(input_tensor)
    conv_layer1 = LeakyReLU(negative_slope=leaky_relu_alpha)(conv_layer1)

    conv_layer1 = Conv2D(conv_filter_num, (4, 1) ,padding="same")(conv_layer1)
    conv_layer1 = LeakyReLU(negative_slope=leaky_relu_alpha)(conv_layer1)

    conv_layer1 = Conv2D(conv_filter_num,(4, 1),padding="same")(conv_layer1)
    conv_layer1 = LeakyReLU(negative_slope=leaky_relu_alpha)(conv_layer1)

    # Conv block 2
    conv_layer2 = Conv2D(conv_filter_num, (1, 2), strides=(1, 2))(conv_layer1)
    conv_layer2 = LeakyReLU(negative_slope=leaky_relu_alpha)(conv_layer2)

    conv_layer2 = Conv2D(conv_filter_num,(4, 1), padding="same")(conv_layer2)
    conv_layer2 = LeakyReLU(negative_slope=leaky_relu_alpha)(conv_layer2)

    conv_layer2 = Conv2D(conv_filter_num, (4, 1), padding="same")(conv_layer2)
    conv_layer2 = LeakyReLU(negative_slope=leaky_relu_alpha)(conv_layer2)

    # Conv block 3
    conv_layer3 = Conv2D(conv_filter_num, (1, 10))(conv_layer2)
    conv_layer3 = LeakyReLU(negative_slope=leaky_relu_alpha)(conv_layer3)

    conv_layer3 = Conv2D(conv_filter_num, (4, 1), padding="same")(conv_layer3)
    conv_layer3 = LeakyReLU(negative_slope=leaky_relu_alpha)(conv_layer3)

    conv_layer3 = Conv2D(conv_filter_num, (4, 1), padding="same")(conv_layer3)
    conv_layer3 = LeakyReLU(negative_slope=leaky_relu_alpha)(conv_layer3)



    # Inception module 1
    inception_module1 = Conv2D(inception_num, (1, 1), padding="same")(conv_layer3)
    inception_module1 = LeakyReLU(negative_slope=leaky_relu_alpha)(inception_module1)

    inception_module1 = Conv2D(inception_num, (3, 1), padding="same")(inception_module1)
    inception_module1 = LeakyReLU(negative_slope=leaky_relu_alpha)(inception_module1)

    # Inception module 2
    inception_module2 = Conv2D(inception_num, (1, 1), padding="same")(conv_layer3)
    inception_module2 = LeakyReLU(negative_slope=leaky_relu_alpha)(inception_module2)

    inception_module2 = Conv2D(inception_num, (5, 1), padding="same")(inception_module2)
    inception_module2 = LeakyReLU(negative_slope=leaky_relu_alpha)(inception_module2)

    # Inception module 3
    inception_module3 = MaxPooling2D((3, 1), strides=(1, 1), padding="same")(conv_layer3)
    
    inception_module3 = Conv2D(inception_num, (1, 1), padding="same")(inception_module3)
    inception_module3 = LeakyReLU(negative_slope=leaky_relu_alpha)(inception_module3)

    # Concatenate + reshape: (batch, T, 1, 3 * inception_num) -> (batch, T, 3 * inception_num)
    inception_module_final = concatenate([inception_module1, inception_module2, inception_module3], axis=-1)
    transformer_input = Reshape((int(inception_module_final.shape[1]), int(inception_module_final.shape[3])))(inception_module_final)
    embedding_dim = int(transformer_input.shape[-1])


    # Positional embedding
    x = PositionalEmbedding(sequence_length=lookback_timestep, embedding_dim=embedding_dim)(transformer_input)

    # Transformer Encoder stack
    for _ in range(transformer_num_layers):
        x = transformer_encoder_block(
            inputs=x,
            num_heads=transformer_num_heads,
            key_dim=transformer_key_dim,
            ff_dim=transformer_ff_dim,
            dropout_rate=dropout_rate,
            leaky_relu_alpha=leaky_relu_alpha
        )

    x = x[:, -1 ,:]
    x = Dense(dense_num)(x)
    x = LeakyReLU(negative_slope=leaky_relu_alpha)(x)
    x = Dropout(dropout_rate)(x)

    model_output = Dense(3, activation="softmax")(x)

    model = Model(inputs=input_tensor, outputs=model_output)

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )

    return model