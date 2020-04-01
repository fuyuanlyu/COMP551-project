### Prepare for training & testing dataset. Define dataset class.
import torch
import torchvision.transforms as transforms
import random
from torch.utils.data import Dataset
from PIL import Image
import matplotlib.pyplot as plt
import pickle

class CIFAR10_from_array(Dataset):
    def __init__(self, data, label, transform=None):
        ##############################################
        ### Initialize paths, transforms, and so on
        ##############################################
        # self.data = torch.from_numpy(data).float()
        # self.label = torch.from_numpy(label).long()
        self.data = data
        self.label = label
        self.transform = transform
        self.img_shape = data.shape

    def __getitem__(self, index):
        ##############################################
        # 1. Read from file (using numpy.fromfile, PIL.Image.open)
        # 2. Preprocess the data (torchvision.Transform).
        # 3. Return the data (e.g. image and label)
        ##############################################

        img = Image.fromarray(self.data[index])
        label = self.label[index]
        if self.transform is not None:
            img = self.transform(img)
        else:
            img_to_tensor = transforms.ToTensor()
            img = img_to_tensor(img)
            # label = torch.from_numpy(label).long()
        return img, label

    def __len__(self):
        ##############################################
        ### Indicate the total size of the dataset
        ##############################################
        return len(self.data)

    def plot_image(self, number):
        file = self.data
        label = self.label
        fig = plt.figure(figsize=(3, 2))
        # img = return_photo(batch_file)
        plt.imshow(file[number])
        plt.title(classes[label[number]])


    def setup_data_aug():
        print("Using real-time data augmentation.\n")
        # This will do preprocessing and realtime data augmentation:
        from keras.preprocessing.image import ImageDataGenerator

        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=0,  # randomly rotate images in the range
            # (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally
            # (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically
            # (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False  # randomly flip images
        )
        return datagen


    # Normalize for R, G, B with img = img - mean / std
    def normalize_dataset(data):
        mean = data.mean(axis=(0,1,2)) / 255.0
        std = data.std(axis=(0,1,2)) / 255.0
        normalize = transforms.Normalize(mean=mean, std=std)
        return normalize

    def transform(normalize_dataset,X_train,X_test):
        train_transform_aug = transforms.Compose([
            transforms.Resize((40, 40)),  # resize the image
            transforms.RandomCrop((32, 32)),  # random crop
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            normalize_dataset(X_train)
        ])

        # Also use X_train in normalize since train/val sets should have same distribution
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize_dataset(X_train)
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize_dataset(X_test)
        ])
    def load_cifar10_data(filename):
        with open('../input/cifar10/'+ filename, 'rb') as file:
            batch = pickle.load(file, encoding='latin1')

        features = batch['data']
        labels = batch['labels']
        return features, labels

    def one_hot_encode(x):
        """
            argument
                - x: a list of labels
            return
                - one hot encoding matrix (number of labels, number of class)
        """
        encoded = np.zeros((len(x), 10))

        for idx, val in enumerate(x):
            encoded[idx][val] = 1

        return encoded
