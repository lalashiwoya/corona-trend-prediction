import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Activation
from argparse import ArgumentParser
from pathlib import Path
from sklearn.metrics import mean_absolute_error, r2_score
import csv
import os
import random
from gooey import Gooey, GooeyParser

def create_dataset(data, time_span, train_length):
    """ Split the dataset into train and test set"""
    train = data[:train_length]
    val = data[train_length:]
    padding_length = train_length - len(val)
    val_padding = np.concatenate((train[-padding_length:], val), axis=0)
    x_train = np.expand_dims(train[:-time_span], axis=1)
    y_train = np.expand_dims(train[time_span:], axis=1)
    x_val = np.expand_dims(val_padding[:-time_span], axis=1)
    y_val = np.expand_dims(val_padding[time_span:], axis=1)
    return x_train, y_train, x_val, y_val


def rmse(predictions, test_data):
    """ Calculate the root mean squared error between the predictions and gold standard"""
    residuals = test_data - predictions
    return np.sqrt(np.mean(residuals ** 2))


def mape(y_true, y_pred):
    """ Calculate the mean absolute percentage error between the predictions and gold standard"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (y_true + 0.01)))


def mae(y_true, y_pred):
    """ Calculate the mean average error between the predictions and gold standard"""
    return mean_absolute_error(y_true, y_pred)


def simple_moving_average(data, window):
    """ Use moving average to smooth the a list of data"""
    return pd.Series(data).rolling(window).mean().iloc[window - 1:].values


def columns_moving_average(data, windows):
    """ Use moving average to smooth colunms of data"""
    data_with_ma = np.zeros((data.shape[0] - windows + 1, data.shape[1]))
    for i in range(data.shape[1]):
        sma = simple_moving_average(data[:, i], windows)
        data_with_ma[:, i] = sma
    return data_with_ma


def zero_mean_unit_variance(data):
    """Normalize the data"""
    normalized_data = data - np.mean(data)  # zero mean
    normalized_data = normalized_data / np.std(normalized_data)  # unit variance
    return normalized_data


def combine_data(arr_1, arr_2):
    """Concatenate the train and test data to create a complete kurve"""
    return np.concatenate((arr_1, arr_2), axis=0)[:, 0]


def simpleLSTM(input_shape):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(32, return_sequences=False, activation='relu', input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(Dense(input_shape[0],activation='relu'))
    model.compile(loss='mae', optimizer='adam')
    return model

@Gooey(program_name="Corona Trend Prediction")
def main():
    parser = GooeyParser(description='In USA, UK and Germany')
    #parser = ArgumentParser()
    parser.add_argument('--time_span', type=int, required=True, help="The length of time series used to train")
    #parser.add_argument('--train_test_split', type=float, default=0.75)
    # parser.add_argument('--dropout',type=float)
    parser.add_argument('--feature_set', required=True,  nargs='+',choices=["vader","flair","UK","USA","Germany","mcx","dji","dax"], widget='Listbox', gooey_options={
                        'height': 50,
                        'text_color': '',
                        'hide_heading': True,
                        'hide_text': True,
                    })
    parser.add_argument('--country', required=True, type=str, choices=['USA','UK','Germany'], widget='Dropdown',
                        help="The country for which you want to make corona prediction")
    #parser.add_argument('--input_dir', type=Path, default='data/', help="The input directory where stores the data")
    parser.add_argument('--input_path', type=Path, required=True,help="The input path where the data file is stored",widget='FileChooser')
    parser.add_argument('--train_test_split', type=float, default=0.7)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=50)

    parser.add_argument('--model_dir', type=Path, default='model/',
                        help="The output directory where the best model will be written")
    parser.add_argument("--output_dir", type=Path, default='experiment_final_output/',
                        help="The output directory where the model predictions will be written")
    parser.add_argument("--record_dir", type=Path, default='experiment_final_record/',
                        help="The output directory where the experiment record will be written")
    parser.add_argument("--moving_average", type=bool, default=True,
                        help="Choose whether you want to smooth the original curve")
    parser.add_argument("--windows", type=int, default=3,
                        help="The length of windows you use to smooth the curve")
    parser.add_argument("--seed_value", type=int, default=0,
                        help="Specify the value of random seed you are going to use during the whole process")

    args = parser.parse_args()
    
    # set the random seed of the system
    os.environ['PYTHONHASHSEED'] = str(args.seed_value)
    random.seed(args.seed_value)
    np.random.seed(args.seed_value)
    tf.random.set_random_seed(args.seed_value)
    session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    tf.compat.v1.keras.backend.set_session(sess)

    feature_dict = {'vader': 0, 'flair': 1, 'UK': 2, 'USA': 3, 'Germany': 4, 'mcx': 5, 'dji': 6, 'dax': 7}
    feature_set = args.feature_set
    stocks = ['dji', 'dax', 'mcx']
    data = pd.read_csv(args.input_path)
    data.columns = feature_dict.keys()
    country = feature_dict[args.country]

    features = list([feature_dict[item] for item in feature_set])

    if country not in features:
        features.append(country)
    else:
        features.remove(country)
        features.append(country)

    data = np.array(data, dtype=np.float32)
    data = data[:, features]

    for stock in stocks:
        if stock in feature_set:
            idx = features.index(feature_dict[stock])
            data[:, idx] = zero_mean_unit_variance(data[:, idx])

    if args.moving_average:
        data_with_ma = columns_moving_average(data, windows=args.windows)
    else:
        data_with_ma = data
    data_with_date = np.column_stack((data_with_ma, np.arange(len(data_with_ma))))

    train_length = int(len(data_with_date) * args.train_test_split)
    val_length = len(data_with_date) - train_length

    x_train, y_train, x_val, y_val = create_dataset(data_with_date, time_span=args.time_span, train_length=train_length)

    input_shape = (1, len(feature_set))
    checkpoint_path = os.path.join(args.model_dir, "best_model.ckpt")
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1,
                                                     monitor='val_loss', save_best_only=True)
    model = simpleLSTM(input_shape=input_shape)
    if args.country in feature_set:
        index=1
    else:
        index=2
    model.fit(x_train[:, :, :-index], y_train[:, :, -2], epochs=args.epochs,
              validation_data=(x_val[:, :, :-index], y_val[:, :, -2]), callbacks=[cp_callback], batch_size=args.batch_size,
              shuffle=False)
    model.load_weights(checkpoint_path)

    pred_train = model.predict(x_train[:, :, :-index])
    pred_train = np.concatenate((pred_train, y_train[:, :, -index]), axis=1)
    pred_val = model.predict(x_val[:, :, :-index])
    pred_val = np.concatenate((pred_val, y_val[:, :, -index]), axis=1)[-val_length:]
    date = np.arange(len(pred_train) + len(pred_val))
    pred_total = combine_data(pred_train, pred_val)
    actual_total = combine_data(y_train[:, 0, -2:], y_val[-val_length:, 0, -2:])
    actual_pred = combine_data(y_train[:, 0, -2:], pred_val)

    root_mse = round(rmse(np.array(pred_total), np.array(actual_total)), 2)
    mean_ape = round(mape(np.array(pred_total), np.array(actual_total)), 2)
    mean_ae = round(mae(np.array(pred_total), np.array(actual_total)), 2)
    r2 = round(r2_score(np.array(pred_total), np.array(actual_total)), 2)
    features = "+".join(feature_set)

    print()
    print(f"Use feature set [{features}] to predict corona trend in {args.country}, the errors between predictions and gold standard are: ")
    print('Root Mean Squared Error: ',root_mse)
    print('Mean Absolute Percentage Error: ',mean_ape)
    print('Mean Absolute Error: ',mean_ae)
    print('R2: ',r2)

    # write the predictions to a csv file
    fname = f"{features},time_span={args.time_span},rmse={root_mse},mape={mean_ape},mae={mean_ae},r2={r2} in {args.country}"
    filepath = os.path.join(args.output_dir, f"{fname}.csv")
    df = pd.DataFrame({'date': date, "actual new cases from train + prediction on the test": actual_pred,
                       "prediction on the train+test set": pred_total,
                       f"actual {args.country} cases": actual_total})
    df.to_csv(filepath, index=False)
    
    # write the errors into a csv file
    record_path = args.record_dir / f"{args.country}.csv"
    # record_path=os.path.join(args.record_dir,f"2 layer RNN in {args.country_to_be_predicted}.csv")
    with open(record_path, 'a+') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([features, args.time_span, root_mse, mean_ape, mean_ae, r2])


if __name__ == "__main__":
    main()
