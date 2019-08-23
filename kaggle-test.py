import datetime
import os
from os import walk

import pandas as pd
from keras_preprocessing.image import ImageDataGenerator, np
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator


class Testing:

    def __init__(self):
        labels = {
            'dog': 1,
            'cat': 0
        }
        self.PATH = 'dataset/test_set'
        self.IMAGE_WIDTH = 128
        self.IMAGE_HEIGHT = 128
        self.IMAGE_SIZE = (self.IMAGE_WIDTH, self.IMAGE_HEIGHT)
        self.IMAGE_CHANNELS = 3
        self.BATCH_SIZE = 15
        self.LABEL_MAP = labels

    def getdataset(self):
        """
        Preparing Testing Data
        """
        files = []
        for (dirpath, dirnames, filenames) in walk(self.PATH):
            for filename in filenames:
                fullfilepath = os.path.join(dirpath, filename)
                files.append(fullfilepath)

        df = pd.DataFrame({
            'filename': files
        })

        nb_samples = df.shape[0]

        return df, nb_samples

    def create_model(self):
        """
        Create a Deep Learning Model
        :return:
        """
        model = Sequential()

        model.add(Conv2D(32, (3, 3), activation='relu',
                         input_shape=(self.IMAGE_WIDTH, self.IMAGE_HEIGHT, self.IMAGE_CHANNELS)))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.25))
        model.add(Dense(2, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        model.summary()

        return model

    def run(self):
        test_df, nb_samples = self.getdataset()
        test_gen = ImageDataGenerator(
            rescale=1. / 255
        )

        test_generator = test_gen.flow_from_dataframe(
            test_df,
            x_col='filename',
            y_col=None,
            class_mode=None,
            target_size=self.IMAGE_SIZE,
            batch_size=self.BATCH_SIZE,
            shuffle=False
        )

        model = self.create_model()
        model.load_weights('model.h5')

        predict = model.predict_generator(
            test_generator,
            steps=np.ceil(nb_samples / self.BATCH_SIZE)
        )

        test_df['category'] = np.argmax(predict, axis=-1)
        test_df['category'] = test_df['category'].replace(self.LABEL_MAP)

        sample_test = test_df.head(18)
        sample_test.head()

        for index, row in sample_test.iterrows():
            print('Predicted {0} and filename is {1}'.format(row['category'], row['filename']))


if __name__ == '__main__':
    testing = Testing()
    startTime = datetime.datetime.now()

    testing.run()

    finishTime = datetime.datetime.now()
    elapsed = finishTime - startTime

    print('Testing Done!!! Time taken = {0}'.format(elapsed))
