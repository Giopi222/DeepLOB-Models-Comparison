from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, LeakyReLU, MaxPooling2D,
    concatenate, LSTM, Reshape, Dense)
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    LeakyReLU,
    MaxPooling2D,
    concatenate,
    Reshape)


'''
DeepLOB model (Zhang, 2018)

Input 100 x 40                         Input(shape=(T, F, 1))

1x2@16 stride 1x2                      Conv2D(16, (1,2), strides=(1,2))
4x1@16                                 Conv2D(16, (4,1), padding="same")
4x1@16                                 Conv2D(16, (4,1), padding="same")

1x2@16 stride 1x2                      Conv2D(16, (1,2), strides=(1,2))
4x1@16                                 Conv2D(16, (4,1), padding="same")
4x1@16                                 Conv2D(16, (4,1), padding="same")

1x10@16                                Conv2D(16, (1,10))
4x1@16                                 Conv2D(16, (4,1), padding="same")
4x1@16                                 Conv2D(16, (4,1), padding="same")

Inception@32                           tre rami con 32 filtri ciascuno

LSTM@64                                LSTM(64)
Dense softmax                          Dense(3, activation="softmax")
'''


def initiate_DeepLOB_model(
    lookback_timestep,
    feature_num,
    conv_filter_num,
    inception_num,
    LSTM_num,
    leaky_relu_alpha,
    loss,
    optimizer,
    metrics):
    

    input_tensor = Input(shape=(lookback_timestep, feature_num, 1))


    # Conv block 1
    conv_layer1 = Conv2D(conv_filter_num, (1, 2), strides=(1, 2))(input_tensor)
    conv_layer1 = LeakyReLU(negative_slope=leaky_relu_alpha)(conv_layer1)

    conv_layer1 = Conv2D(conv_filter_num, (4, 1), padding="same")(conv_layer1)
    conv_layer1 = LeakyReLU(negative_slope=leaky_relu_alpha)(conv_layer1)

    conv_layer1 = Conv2D(conv_filter_num, (4, 1), padding="same")(conv_layer1)
    conv_layer1 = LeakyReLU(negative_slope=leaky_relu_alpha)(conv_layer1)

    # Conv block 2
    conv_layer2 = Conv2D(conv_filter_num, (1, 2), strides=(1, 2))(conv_layer1)
    conv_layer2 = LeakyReLU(negative_slope=leaky_relu_alpha)(conv_layer2)

    conv_layer2 = Conv2D(conv_filter_num, (4, 1), padding="same")(conv_layer2)
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

    # Concatenate inception modules
    inception_module_final = concatenate([inception_module1, inception_module2, inception_module3],axis=-1)
    inception_module_final = Reshape((lookback_timestep, 3 * inception_num))(inception_module_final)



    # LSTM
    LSTM_output = LSTM(LSTM_num)(inception_module_final)

    # Output layer
    model_output = Dense(3, activation="softmax")(LSTM_output)

    DeepLOB_model = Model(inputs=input_tensor, outputs=model_output)

    DeepLOB_model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )

    return DeepLOB_model




