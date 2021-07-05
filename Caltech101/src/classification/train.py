import pickle
import time
from typing import Any, Tuple, List

import torchvision.transforms as transforms
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Subset, DataLoader
from torchvision.transforms import Compose
from torchvision import models

from sklearn.model_selection import train_test_split
from prettytable import PrettyTable

import os
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

# logging.basicConfig(filename="training.log",
#                     format='%(asctime)s %(message)s',
#                     filemode='w')


from models import classification_net as Cnet

mean: List[float] = [0.5, 0.5, 0.5]
stdev: List[float] = [0.5, 0.5, 0.5]


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

    def __init__(self):
        self.data_directory = "/home/harsh.shukla/Caltech101/Data/101_ObjectCategories"
        self.result_directory = "/home/harsh.shukla/Caltech101/results"
        self.batch_size = 32
        self.test_train_split = 0.1
        self.train_transform = Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),  # Center crop to 224
            transforms.ToTensor(),  # Turn PIL Image to torch.Tensor
            transforms.Normalize(mean, stdev)
            # Normalizes tensor with mean and standard deviation
        ])
        self.train_dataset_size = 0
        self.test_dataset_size = 0
        logging.basicConfig(filename=os.path.join(self.result_directory, "training.log"),
                            format='%(asctime)s %(message)s',
                            filemode='w')
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)
        self.NUM_OF_CLASSES = 101
        self.classes = {}
        self.LOG_FREQUENCY = 5
        self.num_epochs = 500
        self.learning_rate = 0.00005

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

    def create_data_loader(self) -> Tuple[DataLoader, DataLoader]:
        image_list: list = []
        label_list: list = []
        label: int = 0
        data: list = list()

        # Read class directories in the form a list.
        directory_list = os.listdir(os.path.join(self.data_directory))
        for classes in directory_list:
            dir_name = os.path.join(self.data_directory, classes)
            temp_image = self.load_images_from_directory(dir_name)
            self.logger.info(f"Loading images from directory - {classes}: size - {len(temp_image)}: DONE")
            image_list.extend(temp_image)
            label_list.extend([label for _ in range(len(temp_image))])
            self.classes[classes] = label
            label = label + 1

        self.logger.info(f"Loading image from directory: DONE")
        image_list = self.create_arrays(image_list)

        for input, target in zip(image_list, label_list):
            data.append([input, target])

        train_test_dict: dict = self.train_val_dataset(data)
        self.train_dataset_size = len(train_test_dict['train'])
        self.test_dataset_size = len(train_test_dict['test'])
        self.logger.info(f"Train-Test Split: DONE")
        trainloader: DataLoader[Any] = DataLoader(dataset=train_test_dict['train'],
                                                  batch_size=self.batch_size,
                                                  shuffle=True,
                                                  )
        testloader: DataLoader[Any] = DataLoader(dataset=train_test_dict['test'],
                                                 batch_size=self.batch_size,
                                                 shuffle=True,
                                                 )

        return trainloader, testloader


class training(pre_processing):

    def __init__(self) -> object:
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
        self.logger.info(f"{table}")
        self.logger.info(f"Total Trainable Params: {total_params}")

    def train_model(self, checkpoint_file: object, loss_criteria, network: object, optimizer: Adam, start,
                    test_loader: DataLoader, train_loader: DataLoader) -> Tuple[list, list]:
        num_of_passes = 0

        # save best config
        best_epoch = 0
        best_val_acc = 0.0

        # save accuracy and loss
        train_accuracies = []
        train_losses = []
        test_accuracies = []
        test_losses = []

        for epoch in range(self.num_epochs):
            running_corrects_train = 0
            running_loss_train = 0.0

            optimizer.zero_grad()

            for input_, labels in train_loader:
                if torch.cuda.is_available():
                    input_ = input_.to(device)
                    labels = labels.to(device)
                output = network(input_)

                _, preds = torch.max(output, 1)
                loss = loss_criteria(output, labels)

                # Log loss
                if num_of_passes % self.LOG_FREQUENCY == 0:
                    self.logger.info('Step {}, Loss {}'.format(num_of_passes, loss.item()))

                loss.backward()
                optimizer.step()

                num_of_passes += 1
                # store loss and accuracy values
                running_corrects_train += torch.sum(preds == labels.data).data.item()
                running_loss_train += loss.item() * input_.size(0)

            train_acc = running_corrects_train / float(self.train_dataset_size)
            train_loss = running_loss_train / float(self.train_dataset_size)
            print(f"Train accuracy: {train_acc}")

            train_accuracies.append(train_acc)
            train_losses.append(train_loss)  # loss computed as the average on mini-batches

            running_corrects_test = 0
            running_loss_test = 0.0

            with torch.set_grad_enabled(False):
                optimizer.zero_grad()
                for test_input, test_labels in test_loader:
                    if torch.cuda.is_available():
                        test_input = test_input.to(device)
                        test_labels = test_labels.to(device)
                    test_output = network(test_input)
                    _, test_preds = torch.max(test_output, 1)

                    test_loss = loss_criteria(test_output, test_labels)

                    running_corrects_test += (test_labels == test_preds).sum().item()
                    running_loss_test += test_loss.item() * test_input.size(0)

                test_acc = running_corrects_test / float(self.test_dataset_size)
                test_loss = running_loss_test / float(self.test_dataset_size)

                test_accuracies.append(test_acc)
                test_losses.append(test_loss)

                self.logger.info(f"epoch:{epoch} - Test Accuracy: {test_acc:.4f}")
                print(f"epoch:{epoch} - Overall Test Accuracy: {test_acc}")

                # Check if the current epoch val accuracy is better than the best found until now
                if test_acc >= best_val_acc:
                    best_val_acc = test_acc
                    best_epoch = epoch

            if epoch % 5 == 0:
                torch.save({
                    'model_state_dict': network.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, checkpoint_file)
        self.logger.info(f"\nBest epoch: {best_epoch + 1}\n{best_val_acc:.4f} (Validation Accuracy)\n")
        self.logger.info(f"> In {(time.time() - start) / 60:.2f} minutes")

        return train_losses, test_losses

    def set_parameter_requires_grad(self, model, feature_extracting: bool):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False

    def initialise_model(self, model_name: str, use_pretrained=True, feature_extract=False):
        model_ft = None
        if model_name == "VGG":
            model_ft = models.vgg19_bn(pretrained=use_pretrained)
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Linear(num_ftrs, self.NUM_OF_CLASSES)
            parameters_to_optimizer = model_ft.classifier.parameters()

        if model_name == "CNet":
            model_ft = Cnet()
            parameters_to_optimizer = model_ft.parameters()
        return model_ft, parameters_to_optimizer

    def train_network(self, train_loader: DataLoader, test_loader: DataLoader):

        # Creating directories for results
        self.logger.info(f"Creating results directory")
        if not os.path.exists(self.result_directory):
            os.makedirs(self.result_directory)

        self.logger.info("Creating Dump directory for keeping track of losses")
        dump_dir = os.path.join(self.result_directory, f"Dump")
        if not os.path.exists(dump_dir):
            os.makedirs(dump_dir)

        self.logger.info("Creating directory for keeping plots and images")
        image_results_dir = os.path.join(self.result_directory, f"images")
        if not os.path.exists(image_results_dir):
            os.makedirs(image_results_dir)

        self.logger.info("Check for pre-existing Dump files")
        if os.path.exists((os.path.join(dump_dir, "classification_train_loss.txt"))):
            train_loss_db = open(os.path.join(dump_dir, "classification_train_loss.txt"), 'rb')
            train_loss_dump = pickle.load(train_loss_db)
        if os.path.exists((os.path.join(dump_dir, "classification_test_loss.txt"))):
            test_loss_db = open(os.path.join(dump_dir, "classification_test_loss.txt"), 'rb')
            test_loss_dump = pickle.load(test_loss_db)

        # Checkpointing
        checkpoints = os.path.join(self.result_directory, "Checkpoints")
        if not os.path.exists(checkpoints):
            os.makedirs(checkpoints)
        checkpoint_file = os.path.join(checkpoints, "check.pt")
        self.logger.info("Directories Created")

        # Initialising Network

        network, parameters = self.initialise_model("CNet")
        network.to(device)

        start = time.time()

        optimizer: Adam = torch.optim.Adam(parameters, lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-8)
        loss_criteria = nn.CrossEntropyLoss()

        self.logger.info(f"Initialised Network, Optimizer and Loss")
        self.logger.info(f"Number of Parameters in Network:{self.count_parameters(network)}")

        # Load checkpoints if exist
        #         if os.path.exists(checkpoint_file):
        #             print("Loading from Previous Checkpoint...")
        #             self.logger.info("Loading from Previous Checkpoint...")
        #             checkpoint = torch.load(checkpoint_file)
        #             network.load_state_dict(checkpoint['model_state_dict'])
        #             optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        #             network.train()
        #         else:
        #             print("No previous checkpoints exist, initialising network from start...")
        #             self.logger.info("No previous checkpoints exist, initialising network from start...")

        # Train Model
        train_losses, test_losses = self.train_model(checkpoint_file, loss_criteria, network, optimizer, start,
                                                     test_loader, train_loader)

        with open(os.path.join(dump_dir, "classification_train_loss.txt"), 'wb') as f:
            pickle.dump(train_losses, f)

        with open(os.path.join(dump_dir, "classification_test_loss.txt"), 'wb') as f:
            pickle.dump(test_losses, f)


if __name__ == '__main__':
    print(f'Started the Data Fetching and Training')
    process = training()
    train_loader, test_loader = process.create_data_loader()
    print(f'Initialised Training for the Network')
    process.train_network(train_loader, test_loader)
