import numpy as np
import matplotlib.pyplot as plt


class Checker:

    def __init__(self, resolution, tile_size):
        # defines the number of pixels in each dimension
        self.resolution = resolution
        # defines the number of pixel an individual tile has in each dimension
        self.tile_size = tile_size

        self.output = self.draw().copy()

    def draw(self):

        # draw a square element 4 * 4 with black and white
        white = np.zeros((2 * self.tile_size, 2 * self.tile_size))
        white[self.tile_size:, 0:self.tile_size] = 1
        white[0:self.tile_size, self.tile_size:] = 1

        # use numpy.tile to cover the block element
        n = int(self.resolution / (2 * self.tile_size))
        image = np.tile(white, (n, n))
        return image

    def show(self):
        img = self.draw()
        plt.imshow(img, cmap='gray', vmin=0, vmax=1)
        plt.show()


class Circle:
    def __init__(self, resolution, radius, position):
        # defines the number of pixels in each dimension
        self.resolution = resolution
        # defines the number of pixel an individual tile has in each dimension
        self.radius = radius

        self.position = position

        self.output = self.draw().copy()

    def draw(self):
        image = np.zeros((self.resolution, self.resolution))  # Input array

        # Size of your array of points image:
        # (image_size_x, image_size_y) = image.shape

        # Disk definition:
        (center_x, center_y) = self.position

        x_grid, y_grid = np.meshgrid(np.arange(image.shape[0]), np.arange(image.shape[1]))

        # Array of booleans with the disk shape
        disk = np.where(((x_grid - center_x) ** 2 + (y_grid - center_y) ** 2) <= self.radius ** 2)
        # You can now do all sorts of things with the mask "disk":

        image[disk] = 1

        return image

    def show(self):
        img = self.draw()
        plt.imshow(img, cmap='gray', vmin=0, vmax=1)
        plt.show()


class Spectrum:
    def __init__(self, resolution):
        # defines the number of pixels in each dimension
        self.resolution = resolution
        # defines the number of pixel an individual tile has in each dimension

        self.output = self.draw().copy()

    def draw(self):
        image = np.zeros(shape=(self.resolution, self.resolution, 3))
        # define a linspace array from 0 to 1
        a = np.expand_dims(np.linspace(0, 1, self.resolution), axis=1)

        image[:, :, 0] = np.tile(a.T, (self.resolution, 1))
        image[:, :, 1] = np.tile(a, (1, self.resolution))
        image[:, :, 2] = np.flip(image[:, :, 0], axis=1)
        return image

    def show(self):
        img = self.draw()
        plt.imshow(img, vmin=0, vmax=1)
        plt.show()

