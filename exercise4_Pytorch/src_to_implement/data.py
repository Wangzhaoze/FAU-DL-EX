import zipfile
import torch
import torchvision as tv
from torch.utils.data import Dataset
from skimage.io import imread
from skimage.color import gray2rgb



train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


class ChallengeDataset(Dataset):
    def __init__(self, data, mode):
        self.data = data
        if mode == "val" or "train":
            self.mode = mode

        self._transforms = tv.transforms.Compose([
                                                  tv.transforms.ToPILImage(),
                                                  tv.transforms.ToTensor(),
                                                  tv.transforms.Normalize(train_mean, train_std),
                                                  tv.transforms.RandomHorizontalFlip(),
                                                  tv.transforms.RandomVerticalFlip(),
                                                  ])
        self.zfile = zipfile.ZipFile('./images.zip', mode="r")

    def __len__(self):
        # return length of data
        return len(self.data)

    def __getitem__(self, index):
        '''
        convert image from grayscale to RGB
        :param index: index of image file in data
        :return:the image and the corresponding label
        '''
        # load zip-file
        file = self.zfile.open(self.data['filename'][index])
        gray_img = imread(file)

        # transform gray-image into RGB
        image = self._transforms(gray2rgb(gray_img))

        # store image and label as torch.tensor
        image = torch.tensor(image)
        label = torch.tensor([self.data['crack'][index], self.data['inactive'][index]])

        return image, label
