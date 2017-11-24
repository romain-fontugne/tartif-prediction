from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential 
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasClassifier

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
	    cols.append(df.shift(i))
	    names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
	    cols.append(df.shift(-i))
	    if i == 0:
		    names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
	    else:
		    names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
	    agg.dropna(inplace=True)
    return agg

class ihrData(object):
    """Retrive and format Internet Health Report data. """

    def __init__(self):
        """Initialize attributes. """

        self.data = None
        self.features = None
        self.fnames = ["data_2017-05_11_AS7922.csv"]
        self.scaler = None

    def getRawData(self):
        """Grab raw data from ihr and put it in a pandas data frame."""

        dataset = []
        for fi in self.fnames:
            dataset.append(pd.read_csv( fi, header=None, names=["timebin", "magnitude", "asn"]))
        
        df = pd.concat(dataset)
        df.index = df.timebin
        del df["timebin"]

        self.data = df

    def computeFeatures(self):
        """Compute features from the raw data. Convert IPs to binary format."""

        filtered = self.data[self.data.asn == 7922]
	del filtered["asn"]
	
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        filtered = self.scaler.fit_transform(filtered)
	self.features = series_to_supervised(filtered, n_in=24)

    def cleanData(self, trainHours=150*24):
        """Filter out features to be ignored in the analysis."""

	# split into train and test sets
	values = self.features.values
	train = values[:trainHours, :]
	test = values[trainHours:, :]
	# split into input and outputs
	self.train_X, self.train_y = train[:, :-1], train[:, -1]
	self.test_X, self.test_y = test[:, :-1], test[:, -1]
	# reshape input to be 3D [samples, timesteps, features]
	self.train_X = self.train_X.reshape((self.train_X.shape[0], 1, self.train_X.shape[1]))
	self.test_X = self.test_X.reshape((self.test_X.shape[0], 1, self.test_X.shape[1]))
	

    def loadData(self):
        """Load raw data and prepare it for analysis.

        return data, label"""

        self.getRawData()
        self.computeFeatures()
        self.cleanData()

        # design network
        model = Sequential()
        # model.add(LSTM(50, input_shape=(self.train_X.shape[1], self.train_X.shape[2])))
        model.add(LSTM(50, input_shape=(self.train_X.shape[1], self.train_X.shape[2]), return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(100, return_sequences=False))# input_shape=(self.train_X.shape[1], self.train_X.shape[2])))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        model.compile(loss='mae', optimizer='adam')
        # fit network
        history = model.fit(self.train_X, self.train_y, epochs=100, batch_size=28, 
                validation_data=(self.test_X, self.test_y), verbose=2, shuffle=False)
        # plot history
        pyplot.figure()
        pyplot.plot(history.history['loss'], label='train')
        pyplot.plot(history.history['val_loss'], label='test')
        pyplot.legend()
        pyplot.title("Loss")
        pyplot.savefig("loss.pdf")

        yhat = model.predict(self.test_X)
        test_X = self.test_X.reshape((self.test_X.shape[0], self.test_X.shape[2]))
        # invert scaling for forecast
        inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
        inv_yhat = self.scaler.inverse_transform(inv_yhat)
        inv_yhat = inv_yhat[:,0]
        # invert scaling for actual
        test_y = self.test_y.reshape((len(self.test_y), 1))
        inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
        inv_y = self.scaler.inverse_transform(inv_y)
        inv_y = inv_y[:,0]
        # calculate RMSE
        rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
        print('Test RMSE: %.3f' % rmse)
