


# -------
import numpy as np

from keras.layers import Input, Dense, Dropout
from keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
# -------

# -------

# --------------------------------------------------------------

class Encoder:

    """
    Class that build an autoencoder. Useful in order to represent fields suck as a velocity field with a reduced dimension.

    :output_dimension: (int) The reduced dimension of the encoded field
    :input_dimension: (int) The dimension of the input field
    :Encoder: (Encoder)
    """

    def __init__(self,output_dimension):

        self.empty = True
        self.autoencoder = None
        self.encoder = None
        self.input_dimension = None
        self.output_dimension = output_dimension

        self.data = []

    def _train(self,field):

        """
        Adds a new field to the training set of the autoencoder, and train it

        :field: (python list) The field

        """

        if self.empty and field is not None:
            print('Initialize the autoencoder with input_dim := ',len(field))
            self.empty = False
            self.input_dimension = len(field)
            self.autoencoder,self.encoder = self._build_autoencoder(self.input_dimension,self.output_dimension)
            self.data.append(list(field))

        elif field is not None:
            field = self._process_field(field,self.input_dimension)
            if field is not None and len(field) == self.input_dimension:
                self.data.append(list(field))
                self._compute_training()


    def _compute_training(self):

        """
        Process the data, and train on it
        """

        scaler = MinMaxScaler()
        X = np.array(self.data)
        X = scaler.fit_transform(self.data)

        X = train_test_split(X, test_size=0.25)
        x_train,x_test = np.array(X[0]),np.array(X[1])

        x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
        x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))


        if len(self.data) < 500:
            epochs,batch_size = 10,16
        elif len(self.data) < 1000:
            epochs,batch_size = 10,64
        else:
            epochs,batch_size = 10,128


        history = self.autoencoder.fit(x_train, x_train,
                epochs=epochs,
                batch_size=batch_size,
                shuffle=True,
                verbose=0,
                validation_data=(x_test, x_test))


        print('AE trained with ',len(self.data),' samples')

    def _get_encoding(self,field):
        """
        Allow to encode a field

        :field: (python list) The field to encode
        :encoding[0]: (python list) The encoded field
        """

        field = self._process_field(field,self.input_dimension)
        if field is not None:
            field = np.array([field])
            field = field.reshape((len(field), np.prod(field.shape[1:])))
            encoding = self.encoder.predict(field)
            return encoding[0]
        else:
            return np.zeros((self.output_dimension,))



    def _build_autoencoder(self,input_dimension,output_dimension):

        input_img = Input(shape=(input_dimension,))

        encoded = Dense(256, activation='relu')(input_img)
        encoded = Dropout(0.4)(encoded)
        encoded = Dense(128, activation='relu')(encoded)
        encoded = Dropout(0.4)(encoded)
        encoded = Dense(output_dimension, activation='sigmoid')(encoded)

        decoded = Dense(128, activation='relu')(encoded)
        decoded = Dropout(0.4)(decoded)
        decoded = Dense(256, activation='relu')(decoded)
        decoded = Dropout(0.4)(decoded)
        decoded = Dense(input_dimension, activation='sigmoid')(decoded)

        autoencoder = Model(input_img, decoded)

        encoder = Model(input_img, encoded)

        autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

        return autoencoder,encoder

    def _process_field(self,field,input_dimension):


        if field is not None and input_dimension is not None:
            if len(field) < input_dimension:
                size_to_add = input_dimension - len(field)
                to_add = np.zeros((size_to_add,))
                return np.append(field,to_add)
            elif len(field) > input_dimension:
                return np.delete(field,np.s_[input_dimension:])
            else:
                return field
        return None
