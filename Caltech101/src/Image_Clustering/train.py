import pickle
import time
from typing import Any, List
from sklearn.cluster import KMeans
import torchvision.transforms as transforms
import torch
from torch import nn
from torch.utils.data import Subset, DataLoader
from torchvision.transforms import Compose
from torchvision import models
from sklearn.decomposition import PCA
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

mean: List[float] = [0.485, 0.456, 0.406]
stdev: List[float] = [0.229, 0.224, 0.225]


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
        self.NUM_OF_PRINCIPAL_COMPONENTS = 3000
        self.NUM_OF_CLUSTERS = 102
        self.test_train_split = 0.1

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

        # for input, target in zip(image_list, label_list):
        #     data.append([input, target])
        train_test_dict: dict = self.train_val_dataset(image_list)
        self.dataset_size = len(train_test_dict['train'])
        loader: DataLoader[Any] = DataLoader(dataset=train_test_dict['train'],
                                             batch_size=1,
                                             shuffle=True,
                                             )
        return loader


class training(pre_processing):

    def __init__(self) -> object:
        super(training, self).__init__()

    def normalize(self, data):
        range = np.max(data) - np.min(data)
        starts_from_zero = data - np.min(data)
        return starts_from_zero / range

    def t_SNE(self, data, labels, name: str, image_results_dir: str):
        tsne = sklearn.manifold.TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
        tsne_results = tsne.fit_transform(data)

        tx = tsne_results[:, 0]
        ty = tsne_results[:, 1]

        # normalise
        tx = self.normalize(tx)
        ty = self.normalize(ty)

        # visualise
        # initialize a matplotlib plot
        fig = plt.figure()
        ax = fig.add_subplot(111)

        # for every class, we'll add a scatter plot separately
        for i in range(self.NUM_OF_CLASSES):
            # find the samples of the current class in the data
            indices = [index for index, l in enumerate(labels) if l == i]

            # extract the coordinates of the points of this class only
            current_tx = np.take(tx, indices)
            current_ty = np.take(ty, indices)

            # convert the class color to matplotlib format
            color = np.array(labels[label], dtype=np.float) / 255

            # add a scatter plot with the corresponding color and label
            ax.scatter(current_tx, current_ty, c=color, label=label)

        # build a legend using the labels we set previously
        ax.legend(loc='best')

        # finally, show the plot
        plt.savefig(os.path.join(image_results_dir, f"tSNE_{name}"))

    def create_clusters(self, feature_extractor: object, dataloader: DataLoader, image_results_dir: str):
        input_features: list = list()
        for input_ in dataloader:
            if torch.cuda.is_available():
                input_ = input_.to(device)
            # extracting features from VGG
            features = feature_extractor(input_)
            input_features.append(torch.flatten(features, 1).cpu().detach().numpy()[0, :])

        input_features = np.asarray(input_features)

        # Performing PCA analysis for both images and features
        pca = PCA(n_components=self.NUM_OF_PRINCIPAL_COMPONENTS)

        # PCA for image features
        pca.fit(input_features)
        print(f"Explained Variance: {sum(pca.explained_variance_ratio_)}")
        dim_reduced_features = pca.transform(input_features)

        # PCA for images

        # t-SNE for both image sets
        # self.t_SNR(dim_reduced_features)

        # kmeans = KMeans(n_clusters=101, random_state=0).fit(dim_reduced_features)
        # print(kmeans.labels_[:100])
        print(f"Running the elbow method")
        # The lists holds the SSE values and Silhoutte Score for each k

        # TODO: T-SNE

        sse: list = list()
        silhoutte_score: list = list()

        logger.info("Running Elbow method to find the minimum SSE for different values of k")

        for k in range(95, self.NUM_OF_CLUSTERS):
            k_means = sklearn.cluster.KMeans(n_clusters=k)
            k_means.fit(dim_reduced_features)
            logger.info(f"k = {k} ; SSE Value: {k_means.inertia_}")
            logger.info(
                f"k = {k} ; Silhoutte Score: {sklearn.metrics.silhouette_score(dim_reduced_features, k_means.labels_)}")
            print(
                f"k = {k}; Silhoutte Score: {sklearn.metrics.silhouette_score(dim_reduced_features, k_means.labels_)}")
            print(f"k = {k}; SSE Value: {k_means.inertia_}")
            sse.append(k_means.inertia_)
            silhoutte_score.append(sklearn.metrics.silhouette_score(dim_reduced_features, k_means.labels_))

        print(f"Minimum SSE- {min(sse)}; Value of k: {sse.index(min(sse)) + 90}")
        print(
            f"Minimum Silhoutte Score- {min(silhoutte_score)}; Value of k: {silhoutte_score.index(min(silhoutte_score)) + 90}")

        print("Saving Graphs for these metrics in the result directory")
        logger.info(f"Saving Graphs for these metrics in the result directory")

        # # Saving plots for Silhouette Coefficient
        # plt.style.use("fivethirtyeight")
        # plt.plot(range(95, 102), silhoutte_score)
        # plt.xticks(range(95, 102))
        # plt.xlabel("Number of Clusters")
        # plt.ylabel("Silhouette Coefficient")
        # plt.savefig(os.path.join(image_results_dir, "Silhouette Coeffecient"))
        #
        # # Saving plot for SSE Coefficient
        # plt.style.use("fivethirtyeight")
        # plt.plot(range(95, 102), sse)
        # plt.xticks(range(95, 102))
        # plt.xlabel("Number of Clusters")
        # plt.ylabel("SSE")
        # plt.savefig(os.path.join(image_results_dir, "SSE"))

    def image_clusters(self, data_loader: DataLoader):

        # Initialising Feature Extractor
        model_ft = models.vgg19_bn(pretrained=True)
        modules = list(model_ft.children())[:-1]
        model_ft = nn.Sequential(*modules)
        model_ft.eval()
        model_ft.to(device)

        start = time.time()
        self.create_clusters(model_ft, data_loader)
        logger.info(f"Completed the Clustering")


if __name__ == '__main__':
    print(f'Started the Data Fetching and Training')
    process = training()
    data_loader = process.create_data_loader()
    print(f'Starting Image Clustering...')
    process.image_clusters(data_loader)
