import datetime
import os
from os import walk

import pandas as pd
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


class Training:

    def __init__(self, fastrun):
        self.FAST_RUN = fastrun
        self.IMAGE_WIDTH = 128
        self.IMAGE_HEIGHT = 128
        self.IMAGE_SIZE = (self.IMAGE_WIDTH, self.IMAGE_HEIGHT)
        self.IMAGE_CHANNELS = 3
        self.PATH = 'dataset/training_set'
        self.BATCH_SIZE = 15

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

    @staticmethod
    def create_callbacks():
        """
        Create a List of Callbacks
        :return: a list of callbacks
        """
        earlystop = EarlyStopping(patience=18)

        learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                                    patience=2,
                                                    verbose=1,
                                                    factor=0.5,
                                                    min_lr=0.00001)

        callbacks = [earlystop, learning_rate_reduction]

        return callbacks

    def run(self):
        files = []
        categories = []
        for (dirpath, dirnames, filenames) in walk(self.PATH):
            for filename in filenames:
                category = filename.split('.')[0]
                fullfilepath = os.path.join(dirpath, filename)
                files.append(fullfilepath)
                if category == 'dog':
                    categories.append(1)
                else:
                    categories.append(0)

        df = pd.DataFrame({
            'filename': files,
            'category': categories
        })

        # Adjuste Data
        df['category'] = df['category'].replace({0: 'cat', 1: "dog"})

        # Split Data
        train_df, validate_df = train_test_split(df, test_size=0.20, random_state=42)
        train_df = train_df.reset_index(drop=True)
        validate_df = validate_df.reset_index(drop=True)

        total_train = train_df.shape[0]
        total_validate = validate_df.shape[0]

        # Data Augmentation for Training
        train_datagen = ImageDataGenerator(
            rotation_range=15,
            rescale=1. / 255,
            shear_range=0.1,
            zoom_range=0.2,
            horizontal_flip=True,
            width_shift_range=0.1,
            height_shift_range=0.1
        )

        train_generator = train_datagen.flow_from_dataframe(
            train_df,
            x_col='filename',
            y_col='category',
            target_size=self.IMAGE_SIZE,
            class_mode='categorical',
            batch_size=self.BATCH_SIZE
        )

        # Data Augmentation for Validation
        validate_datagen = ImageDataGenerator(
            rescale=1. / 255
        )

        validate_generator = validate_datagen.flow_from_dataframe(
            validate_df,
            x_col='filename',
            y_col='category',
            target_size=self.IMAGE_SIZE,
            class_mode='categorical',
            batch_size=self.BATCH_SIZE
        )

        epochs = 3 if self.FAST_RUN else 50

        model = self.create_model()

        history = model.fit_generator(
            train_generator,
            epochs=epochs,
            validation_data=validate_generator,
            validation_steps=total_validate // self.BATCH_SIZE,
            steps_per_epoch=total_train // self.BATCH_SIZE,
            callbacks=self.create_callbacks()
        )

        model.save_weights('model.h5')


if __name__ == '__main__':
    try:
        training = Training(fastrun=True)
        startTime = datetime.datetime.now()

        training.run()

        finishTime = datetime.datetime.now()
        elapsed = finishTime - startTime

        print('Training Done!!! Time taken = {0}'.format(elapsed))

    except:
        print('An error occurred. Please Check!!')
