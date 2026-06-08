import numpy as np
from tensorflow.keras.utils import to_categorical

'''
dataset: FI2010 (NoAuction --> ZScore)
data source: https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649/data
- train set: CF_7
- test set:  CF_7, CF_8, CF_9
'''


def extract_x_y_data(data, timestamp_per_sample=100, max_samples=None):
    data_x = data[:40, :].T.astype("float32") # LOB features (P_ask, V_ask, P_bid, V_bid)
    data_y = data[-5:, :].T                   # horizons (k = 10, 20, 30, 50, 100)

    N, P_x = data_x.shape
    n_samples = N - timestamp_per_sample + 1

    if max_samples is not None:
        n_samples = min(n_samples, max_samples)

    x = np.zeros((n_samples, timestamp_per_sample, P_x, 1), dtype="float32")

    for i in range(n_samples):
        x[i, :, :, 0] = data_x[i:i + timestamp_per_sample, :]

    y = data_y[timestamp_per_sample - 1:timestamp_per_sample - 1 + n_samples]
    y = y[:, 3].astype(int) - 1
    y = to_categorical(y, num_classes=3).astype("float32")

    return x, y


def load_datasets(base_path, timestamp_per_sample=100):
    train_fi = np.loadtxt(base_path / "Train_Dst_NoAuction_ZScore_CF_7.txt", dtype="float32")

    test_files = [
        "Test_Dst_NoAuction_ZScore_CF_7.txt",
        "Test_Dst_NoAuction_ZScore_CF_8.txt",
        "Test_Dst_NoAuction_ZScore_CF_9.txt",
    ]

    test_arrays = [
        np.loadtxt(base_path / f, dtype="float32")
        for f in test_files
    ]

    test_fi = np.hstack(test_arrays)

    train_x, train_y = extract_x_y_data(train_fi, timestamp_per_sample) 
    test_x, test_y = extract_x_y_data(test_fi, timestamp_per_sample)


    n_train = int(0.8 * len(train_x))

    x_train = train_x[:n_train]
    y_train = train_y[:n_train]

    x_val = train_x[n_train:]
    y_val = train_y[n_train:]

    return x_train, y_train, x_val, y_val, test_x, test_y



## example usage:

#base_path = Path(...)
#x_train, y_train, x_val, y_val, x_test, y_test = load_datasets(base_path=base_path, timestamp_per_sample=100)
#print(x_train.shape, y_train.shape)
#print(x_val.shape, y_val.shape)
#print(x_test.shape, y_test.shape)
