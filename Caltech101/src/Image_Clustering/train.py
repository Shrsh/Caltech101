import pickle
import time
from typing import Any, List

import torchvision.transforms as transforms
import torch
from torch.utils.data import Subset, DataLoader
from torchvision.transforms import Compose
from torchvision import models

from sklearn.model_selection import train_test_split
import sklearn
import matplotlib.pyplot as plt

import os
import shutil
from PIL import Image
import numpy as np
import logging
import warnings

use_cuda = torch.cuda.is_available()
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device_ids = [i for i in range(torch.cuda.device_count())]
device = 'cuda' if use_cuda else 'cpu'
torch.backends.cudnn.benchmark = True
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

logging.basicConfig(filename="training.log",
                    format='%(asctime)s %(message)s',
                    filemode='w')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

mean: List[float] = [0.5, 0.5, 0.5]
stdev: List[float] = [0.5, 0.5, 0.5]


class pre_processing:
    """

    Data Fetching and pre_processing
    """
    train_transform: Compose
    train_test_split: float
    data_directory: str
    dataset_size: int
    test_dataset_size: int
    classes: dict
    result_directory: str
    NUM_OF_CLASSES: int
    LOG_FREQUENCY: int
    NUM_OF_PRINCIPAL_COMPONENTS: int
    NUM_OF_CLUSTERS: int

    def __init__(self):
        self.data_directory = "/home/harsh.shukla/Caltech101/Data/101_ObjectCategories"
        self.result_directory = "/home/harsh.shukla/Caltech101/results"
        self.train_transform = Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),  # Center crop to 224
            transforms.ToTensor(),  # Turn PIL Image to torch.Tensor
            transforms.Normalize(mean, stdev)
            # Normalizes tensor with mean and standard deviation
        ])
        self.dataset_size = 0
        self.NUM_OF_CLASSES = 101
        self.classes = {}
        self.LOG_FREQUENCY = 5
        self.NUM_OF_PRINCIPAL_COMPONENTS = 50
        self.NUM_OF_CLUSTERS = 100

    def train_val_dataset(self, dataset):
        train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=self.test_train_split)
        datasets: dict = {'train': Subset(dataset, train_idx), 'test': Subset(dataset, val_idx)}
        return datasets

    def center_crop_image(self, img, dim):
        width, height = img.shape[1], img.shape[0]
        crop_width = dim[0] if dim[0] < img.shape[1] else img.shape[1]
        crop_height = dim[1] if dim[1] < img.shape[0] else img.shape[0]
        mid_x, mid_y = int(width / 2), int(height / 2)
        cw2, ch2 = int(crop_width / 2), int(crop_height / 2)
        crop_img = img[mid_y - ch2:mid_y + ch2, mid_x - cw2:mid_x + cw2]
        return crop_img

    def load_images_from_directory(self, folder: str) -> list:
        images: list = list()
        list_name: list = list()
        filename: str
        for filename in os.listdir(folder):
            list_name.append(os.path.join(folder, filename))
        list_name.sort()
        for filename in list_name:
            img = Image.open(filename).convert('RGB')
            img = self.train_transform(img)
            if img is not None:
                images.append(img.numpy())
        return images

    def create_arrays(self, data_array):
        data = np.array(data_array)
        data = np.moveaxis(data, 1, -1)
        data = np.moveaxis(data, 1, -1)
        data = data.astype(np.float32)
        return data_array

    def create_data_loader(self) -> DataLoader:
        image_list: list = []
        label_list: list = []
        label: int = 0
        data: list = list()

        # Read class directories in the form a list.
        directory_list = os.listdir(os.path.join(self.data_directory))
        for classes in directory_list:
            dir_name = os.path.join(self.data_directory, classes)
            temp_image = self.load_images_from_directory(dir_name)
            logger.info(f"Loading images from directory - {classes}: size - {len(temp_image)}: DONE")
            image_list.extend(temp_image)
            label_list.extend([label for _ in range(len(temp_image))])
            self.classes[classes] = label
            label = label + 1

        logger.info(f"Loading image from directory: DONE")
        image_list = self.create_arrays(image_list)

        for input, target in zip(image_list, label_list):
            data.append([input, target])

        self.dataset_size = len(data)
        loader: DataLoader[Any] = DataLoader(dataset=data,
                                             batch_size=self.dataset_size,
                                             shuffle=True,
                                             )
        return loader


class training(pre_processing):

    def __init__(self) -> object:
        super(training, self).__init__()

    def create_clusters(self, feature_extractor : object, dataloader: DataLoader, image_results_dir: str):

        for input, labels in dataloader:
            if torch.cuda.is_available():
                input = input.to(device)
            # extracting features from VGG
            features = feature_extractor(input)

            # Performing PCA analysis
            pca = sklearn.decomposition.PCA(n_components=self.NUM_OF_PRINCIPAL_COMPONENTS)
            pca.fit(torch.flatten(features, 1).cpu().numpy())
            dim_reduced_features = pca.transform(torch.flatten(features, 1).cpu().numpy())

            # Performing clustering using K-Means Algorithm.
            print(f"Running the elbow method")
            # The lists holds the SSE values and Silhoutte Score for each k

            sse: list = list()
            silhoutte_score: list = list()

            logger.info("Running Elbow method to find the minimum SSE for different values of k")

            for k in range(90, self.NUM_OF_CLUSTERS):
                k_means = sklearn.cluster.KMeans(n_clusters=k)
                k_means.fit(dim_reduced_features)
                logger.info(f"k = {k} ; SSE Value: {k_means.inertia_}")
                logger.info(f"k = {k} ; Silhoutte Score: {sklearn.metrics.silhouette_score(dim_reduced_features, k_means.labels_)}")
                print(f"k = {k}; Silhoutte Score: {sklearn.metrics.silhouette_score(dim_reduced_features, k_means.labels_)}")
                print(f"k = {k}; SSE Value: {k_means.inertia_}")
                sse.append(k_means.inertia_)
                silhoutte_score.append(sklearn.metrics.silhouette_score(dim_reduced_features, k_means.labels_))

            print(f"Minimum SSE- {min(sse)}; Value of k: {sse.index(min(sse)) + 90}")
            print(f"Minimum Silhoutte Score- {min(silhoutte_score)}; Value of k: {silhoutte_score.index(min(silhoutte_score)) + 90}")

            print("Saving Graphs for these metrics in the result directory")
            logger.info(f"Saving Graphs for these metrics in the result directory")

            # Saving plots for Silhouette Coefficient
            plt.style.use("fivethirtyeight")
            plt.plot(range(90, 101), silhoutte_score)
            plt.xticks(range(90, 101))
            plt.xlabel("Number of Clusters")
            plt.ylabel("Silhouette Coefficient")
            plt.savefig(os.path.join(image_results_dir, "Silhouette Coeffecient"))

            # Saving plot for SSE Coefficient
            plt.style.use("fivethirtyeight")
            plt.plot(range(90, 101), sse)
            plt.xticks(range(90, 101))
            plt.xlabel("Number of Clusters")
            plt.ylabel("SSE")
            plt.savefig(os.path.join(image_results_dir, "SSE"))

    def image_clusters(self, data_loader: DataLoader):

        # Creating directories for results
        logger.info(f"Creating results directory")
        if os.path.exists(self.result_directory):
            shutil.rmtree(self.result_directory, ignore_errors=True)
        else:
            os.mkdir(self.result_directory)

        logger.info("Creating Dump directory for keeping track of losses")
        dump_dir = os.path.join(self.result_directory, f"Dump")
        if os.path.exists(dump_dir):
            shutil.rmtree(dump_dir, ignore_errors=True)
        else:
            os.mkdir(dump_dir)

        logger.info("Creating directory for keeping plots and images")
        image_results_dir = os.path.join(self.result_directory, f"images")
        if os.path.exists(image_results_dir):
            shutil.rmtree(image_results_dir, ignore_errors=True)
        else:
            os.mkdir(image_results_dir)
        logger.info("Directories Created")

        # Initialising Feature Extractor
        network = models.vgg11(pretrained=True).features()
        network.eval()
        network.to(device)

        start = time.time()
        self.create_clusters(network, data_loader, image_results_dir)
        logger.info(f"Completed the Clustering")


if __name__ == '__main__':
    print(f'Started the Data Fetching and Training')
    process = training()
    data_loader = process.create_data_loader()
    print(f'Starting Image Clustering...')
    process.image_clusters(data_loader)