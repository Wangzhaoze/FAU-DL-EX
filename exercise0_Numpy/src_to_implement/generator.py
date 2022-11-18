import os.path
import json
import random

import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
import skimage

from skimage import transform


# In this exercise task you will implement an image generator. Generator objects in python are defined as having a
# next function. This next function returns the next generated object. In our case it returns the input of a neural
# network each time it gets called. This input consists of a batch of images and its corresponding labels.


class ImageGenerator:
    def __init__(self, file_path, label_path, batch_size, image_size, rotation=False, mirroring=False, shuffle=False):
        # Define all members of your generator class object as global members here.
        # These need to include:
        # the batch size
        # the image size
        # flags for different augmentations and whether the data should be shuffled for each epoch
        # Also depending on the size of your data-set you can consider loading all images into memory here already.
        # The labels are stored in json format and can be directly loaded as dictionary.
        # Note that the file names correspond to the dicts of the label dictionary.

        # the path to the directory containing all images file_path as a string
        self.file_path = file_path

        # the path to the JSON file label_path containing the labels again as string

        self.label_path = label_path
        # defining the number of images in a batch.
        self.batch_size = batch_size

        # a list of integers defining the desired image size [height,width,channel]
        self.image_size = image_size

        # and optional bool flags rotation, mirroring, shuffle which default to False
        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle = shuffle
        self.count = 0
        self.data_remainder = 100
        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}

        # TODO: implement constructor


    def next(self):
        # This function creates a batch of images and corresponding labels and returns them.
        # In this context a "batch" of images just means a bunch, say 10 images that are forwarded at once.
        # Note that your amount of total data might not be divisible without remainder with the batch_size.
        # Think about how to handle such cases
        # TODO: implement next method

        # read the json file
        with open(self.label_path, 'rb') as json_file:
            dataset = json.load(json_file)
        json_file.close()

        # shuffle
        # At first shuffle the list of keys , then find the corresponding items and return as new_dict
        if self.shuffle:
            dict_keys = list(dataset.keys())
            random.shuffle(dict_keys)
            new_dict = {}
            for key in dict_keys:
                new_dict[key] = dataset.get(key)
            dataset = new_dict

        # create list of keys and values
        dataset_keys = list(dataset.keys())
        dataset_values = list(dataset.values())

        batch_keys = np.zeros(self.batch_size).astype(np.str)
        batch_values = np.zeros(self.batch_size)
        start = 100 - self.data_remainder

        # if the remainder amount of data is enough for a new batch
        if self.data_remainder >= self.batch_size:

            batch_keys = dataset_keys[start:(start + self.batch_size)]
            batch_values = dataset_values[start:(start + self.batch_size)]

        # if the amount isn't enough
        else:
            # update the current epoch
            self.count = self.count + 1

            # if the remainder is just 0
            if start == 100:
                batch_keys[0:self.batch_size] = dataset_keys[0:self.batch_size]
                batch_values[0:self.batch_size] = dataset_values[0:self.batch_size]

            # if the remainder isn't 0
            else:
                batch_keys[0:self.data_remainder] = dataset_keys[start:100]
                batch_keys[self.data_remainder:self.batch_size] = dataset_keys[0:(
                        self.batch_size - self.data_remainder)]

                batch_values[0:self.data_remainder] = dataset_values[start:100]
                batch_values[self.data_remainder:self.batch_size] = dataset_values[0:(
                        self.batch_size - self.data_remainder)]



        # after batching update the parameter data_remainder
        if self.data_remainder >= self.batch_size:
            self.data_remainder = self.data_remainder - self.batch_size
        else:
            self.data_remainder = self.data_remainder - self.batch_size + 100

        # get the list of labels
        labels = batch_values

        # define a set of images and broadcasting after augment and resize
        images = np.zeros((self.batch_size, self.image_size[0], self.image_size[1], self.image_size[2]))
        for i in range(self.batch_size):
            images[i, :, :, :] = self.augment(
                skimage.transform.resize(np.load(self.file_path + batch_keys[i] + '.npy'), self.image_size))

        return images, labels

    def augment(self, img):
        # this function takes a single image as an input and performs a random transformation
        # (mirroring and/or rotation) on it and outputs the transformed image
        # TODO: implement augmentation function
        # rotation: Y or N
        if self.rotation:
            img = np.rot90(img, k=random.randint(1, 3))
        # mirroring: Y or N
        if self.mirroring:
            img = np.flip(img, axis=random.randint(0, 1))

        return img

    def current_epoch(self):
        # return the current epoch number
        return self.count

    def class_name(self, x):
        # This function returns the class name for a specific input
        # TODO: implement class name function

        # input x is one label, which is returned by function 'next()'
        # class name and label is stored as key-value in dictionary 'self.class_dict'
        name = list(self.class_dict.values())[int(x)]

        return name

    @property
    def show(self):
        """# In order to verify that the generator creates batches as required, this functions calls next to get a
        # batch of images and labels and visualizes it.
        #TODO: implement show method"""

        # define a figure with 3 columns and some rows
        fig = plt.figure(figsize=(int(np.ceil(self.batch_size/3)), 3))

        # show subplots image for example size of batch is 12
        for i in range(self.batch_size):
            images, labels = self.next()
            img = images[i]
            fig.add_subplot(int(np.ceil(self.batch_size/3)), 3, i + 1)

            #adjust pictures and blank
            fig.tight_layout()
            plt.subplots_adjust(wspace=1, hspace=1)

            # get the corresponding class_name by using function class_name()
            title = self.class_name(labels[i])
            plt.title(title, fontdict={'weight': 'normal', 'size': 8})
            plt.axis('off')
            plt.imshow(img)

        fig.show()

        pass


