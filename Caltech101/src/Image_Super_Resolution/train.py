import pickle
import time
from typing import Any, Tuple, List
import math

import torchvision.transforms as transforms
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Subset, DataLoader
from torchvision.transforms import Compose
from models import arch
from sklearn.model_selection import train_test_split
from prettytable import PrettyTable
from torch.nn import init
from initializer import kaiming_normal_

import os
from PIL import Image
import numpy as np
import logging
import yaml
import warnings

use_cuda = torch.cuda.is_available()
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device_ids = [i for i in range(torch.cuda.device_count())]
device = 'cuda' if use_cuda else 'cpu'
torch.backends.cudnn.benchmark = True
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

logging.basicConfig(filename=os.path.join("training.log"),
                    format='%(asctime)s %(message)s',
                    filemode='w')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

mean: List[float] = [0.5, 0.5, 0.5]
stdev: List[float] = [0.5, 0.5, 0.5]

# folder to load config file
CONFIG_PATH = "../"


# Function to load yaml configuration file
def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)

    return config


config = load_config("SuperResolution.yaml")

def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        kaiming_normal_(m.weight.data, mode='fan_in', nonlinearity='leaky_relu')
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)


class pre_processing:
    """

    Data Fetching and pre_processing
    """
    train_transform: Compose
    batch_size: int
    train_test_split: float
    data_directory: str
    train_dataset_size: int
    test_dataset_size: int
    classes: dict
    result_directory: str
    logger: logging
    NUM_OF_CLASSES: int
    LOG_FREQUENCY: int
    num_epochs: int
    learning_rate: float
    negative_slope: float

    def __init__(self):
        self.data_directory = config['dataset']['path']
        self.result_directory = config['results']['result_dir']
        self.batch_size = config['training']['batch_size']
        self.test_train_split = config['training']['test_train_split']
        self.ground_truth_transform = Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),  # Center crop to 224
            transforms.ToTensor(),  # Turn PIL Image to torch.Tensor
            transforms.Normalize(mean, stdev)
            # Normalizes tensor with mean and standard deviation
        ])
        self.input_transform = Compose([
            transforms.Resize(56),
            transforms.CenterCrop(56),  # Center crop to 224
            transforms.ToTensor(),  # Turn PIL Image to torch.Tensor
            transforms.Normalize(mean, stdev)
            # Normalizes tensor with mean and standard deviation
        ])
        self.train_dataset_size = 0
        self.test_dataset_size = 0
        self.LOG_FREQUENCY = 500
        self.num_epochs = config['training']['num_of_epochs']
        self.learning_rate = config['optimizer']['initial_lr']

        # Model Hyperparameters
        self.negative_slope = config['model']['negative_slope']


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

    def load_images_from_directory(self, folder: str, image_type: str) -> list:
        images: list = list()
        list_name: list = list()
        filename: str
        for filename in os.listdir(folder):
            list_name.append(os.path.join(folder, filename))
        list_name.sort()
        for filename in list_name:
            img = Image.open(filename).convert('RGB')
            if image_type == "label":
                img = self.ground_truth_transform(img)
            else:
                img = self.input_transform(img)
            if img is not None:
                images.append(img.numpy())
        return images

    def create_arrays(self, data_array):
        data = np.array(data_array)
        data = np.moveaxis(data, 1, -1)
        data = np.moveaxis(data, 1, -1)
        data = data.astype(np.float32)
        return data_array

    def create_data_loader(self) -> Tuple[DataLoader, DataLoader]:
        data: list = list()

        input_dir_name: str = os.path.join(self.data_directory, "x")
        target_dir_name: str = os.path.join(self.data_directory, "y")
        input_images: list = self.load_images_from_directory(input_dir_name, "input")
        target_images: list = self.load_images_from_directory(target_dir_name, "label")

        logger.info(f"Loading image from directory: DONE")
        # input_images = self.create_arrays(input_images)
        # target_images = self.create_arrays(target_images)

        for input, target in zip(input_images, target_images):
            data.append([input, target])

        train_test_dict: dict = self.train_val_dataset(data)
        self.train_dataset_size = len(train_test_dict['train'])
        self.test_dataset_size = len(train_test_dict['test'])
        logger.info(f"Train-Test Split: DONE")
        trainloader: DataLoader[Any] = DataLoader(dataset=train_test_dict['train'],
                                                  batch_size=self.batch_size,
                                                  shuffle=True,
                                                  )
        testloader: DataLoader[Any] = DataLoader(dataset=train_test_dict['test'],
                                                 batch_size=self.batch_size,
                                                 shuffle=False,
                                                 )

        return trainloader, testloader


class training(pre_processing):

    def __init__(self):
        super(training, self).__init__()

    def count_parameters(self, model: object):
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            param = parameter.numel()
            table.add_row([name, param])
            total_params += param
        logger.info(f"{table}")
        logger.info(f"Total Trainable Params: {total_params}")

    def train_model(self, checkpoint_file: object, loss_criteria, network: object, optimizer: Adam, start,
                    test_loader: DataLoader, train_loader: DataLoader) -> Tuple[list, list]:
        num_of_passes = 0

        # save accuracy and loss
        train_losses = []
        test_losses = []

        for epoch in range(self.num_epochs):
            running_loss_train = 0.0

            optimizer.zero_grad()

            for input_, labels in train_loader:
                if torch.cuda.is_available():
                    input_ = input_.to(device)
                    labels = labels.to(device)
                output = network(input_)
                loss = loss_criteria(output, labels)
                loss.backward()
                optimizer.step()

                num_of_passes += 1
                # store loss and accuracy values
                running_loss_train += loss.item() * input_.size(0)

            train_loss = running_loss_train / float(self.train_dataset_size)
            print(f"Train Loss: {train_loss}")

            train_losses.append(train_loss)  # loss computed as the average on mini-batches

            running_corrects_test = 0
            running_loss_test = 0.0

            with torch.set_grad_enabled(False):
                optimizer.zero_grad()
                for test_input, test_labels in test_loader:
                    if torch.cuda.is_available():
                        test_input = test_input.to(device)
                        test_labels = test_labels.to(device)

                    # Bicubic Testing - Required only once
                    if num_of_passes < 4000:
                        bicubic = torch.nn.Upsample(scale_factor=4)
                        bicubic_output = bicubic(test_input)
                        bi_test_loss = loss_criteria(bicubic_output, test_labels)
                        logger.info(f"Bicubic Test Loss: {bi_test_loss}")
                        logger.info(f"Bicubic PSNR: {10 * math.log(4 / (bi_test_loss * bi_test_loss))}")

                    test_output = network(test_input)

                    test_loss = loss_criteria(test_output, test_labels)
                    # logger.info(f"epoch:{epoch}: PSNR:{10*math.log(4/(bi_test_loss*bi_test_loss))}")
                    # print(f"epoch:{epoch}: PSNR:{10*math.log(4/(bi_test_loss*bi_test_loss))}")

                    running_loss_test += test_loss.item() * test_input.size(0)

                test_loss = running_loss_test / float(self.test_dataset_size)

                test_losses.append(test_loss)

                logger.info(f"epoch:{epoch} - Test MSE Loss: {test_loss:.4f}")

            if epoch % 5 == 0:
                torch.save({
                    'model_state_dict': network.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, checkpoint_file)
        logger.info(f"> In {(time.time() - start) / 60:.2f} minutes")

        return train_losses, test_losses

    def set_parameter_requires_grad(self, model, feature_extracting: bool):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False

    def train_network(self, train_loader: DataLoader, test_loader: DataLoader):

        # Creating directories for results
        logger.info(f"Creating results directory")
        if not os.path.exists(self.result_directory):
            os.makedirs(self.result_directory)

        logger.info("Creating Dump directory for keeping track of losses")
        dump_dir = os.path.join(self.result_directory, f"Dump")
        if not os.path.exists(dump_dir):
            os.makedirs(dump_dir)

        logger.info("Creating directory for keeping plots and images")
        image_results_dir = os.path.join(self.result_directory, f"images")
        if not os.path.exists(image_results_dir):
            os.makedirs(image_results_dir)

        logger.info("Check for pre-existing Dump files")
        if os.path.exists((os.path.join(dump_dir, "train_loss.txt"))):
            train_loss_db = open(os.path.join(dump_dir, "train_loss.txt"), 'rb')
            train_loss_dump = pickle.load(train_loss_db)
        if os.path.exists((os.path.join(dump_dir, "test_loss.txt"))):
            test_loss_db = open(os.path.join(dump_dir, "test_loss.txt"), 'rb')
            test_loss_dump = pickle.load(test_loss_db)

        # Checkpointing
        checkpoints = os.path.join("Checkpoints")
        if not os.path.exists(checkpoints):
            os.makedirs(checkpoints)
        checkpoint_file = os.path.join(checkpoints, "check.pt")
        logger.info("Directories Created")

        # Initialising Network

        network = arch()
        network.to(device)
        network.apply(weight_init)
        start = time.time()

        optimizer: Adam = torch.optim.Adam(network.parameters(), lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-8)
        loss_criteria = nn.MSELoss()

        logger.info(f"Initialised Network, Optimizer and Loss")
        logger.info(f"Number of Parameters in Network:{self.count_parameters(network)}")

        # Load checkpoints if exist
        if os.path.exists(checkpoint_file):
            print("Loading from Previous Checkpoint...")
            logger.info("Loading from Previous Checkpoint...")
            checkpoint = torch.load(checkpoint_file)
            network.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            network.train()
        else:
            print("No previous checkpoints exist, initialising network from start...")
            logger.info("No previous checkpoints exist, initialising network from start...")

        # Train Model
        train_losses, test_losses = self.train_model(checkpoint_file, loss_criteria, network, optimizer, start,
                                                     test_loader, train_loader)

        with open(os.path.join(dump_dir, "train_loss.txt"), 'wb') as f:
            pickle.dump(train_losses, f)

        with open(os.path.join(dump_dir, "test_loss.txt"), 'wb') as f:
            pickle.dump(test_losses, f)


if __name__ == '__main__':
    print(f'Started the Data Fetching and Training')
    process = training()
    train_loader, test_loader = process.create_data_loader()
    print(f'Initialised Training for the Network')
    process.train_network(train_loader, test_loader)
