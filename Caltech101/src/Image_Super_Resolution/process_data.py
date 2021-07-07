import os
from PIL import Image
from torchvision.transforms import Compose
import torchvision.transforms as transforms
import logging
import shutil

mean = [0.5, 0.5, 0.5]
stdev = [0.5, 0.5, 0.5]

logging.basicConfig(filename="Data_Processing.log",
                    format='%(asctime)s %(message)s',
                    filemode='w')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class process_data:
    test_path: str
    train_path: str
    resolution: int
    raw_data: str
    process_data: str
    x: str
    y: str

    def __init__(self, data_directory: str, processed_data_directory: str):
        self.resolution = 4  # Preparing for 4x super resolution.
        self.count = 0
        self.raw_data = data_directory
        self.processed_data = processed_data_directory
        self.x = os.path.join(processed_data_directory, "x")
        self.y = os.path.join(processed_data_directory, "y")
        self.train_transform = Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, stdev),
            transforms.ToPILImage(),
        ])

    def load_images_from_directory(self, folder: str) -> list:
        images: list = list()
        list_name: list = list()
        filename: str

        for filename in os.listdir(folder):
            list_name.append(os.path.join(folder, filename))
        list_name.sort()
        for filename in list_name:
            img = Image.open(filename).convert('RGB')

            # Applying transforms on the images
            img = self.train_transform(img)

            if img is not None:
                images.append(img)
        return images

    def process(self):
        self.count = 0
        image_list: list = list()

        logger.info(f"Creating processsed images directory")
        if os.path.exists(self.processed_data):
            shutil.rmtree(self.processed_data, ignore_errors=True)
        os.mkdir(self.processed_data)

        logger.info(f"Creating directory for inputs")
        if os.path.exists(self.x):
            shutil.rmtree(self.x, ignore_errors=True)
        os.mkdir(self.x)

        logger.info(f"Creating directory for ground truth")
        if os.path.exists(self.y):
            shutil.rmtree(self.y, ignore_errors=True)
        os.mkdir(self.y)

        # List directory under data directory
        directory_list = os.listdir(os.path.join(self.raw_data))
        for classes in directory_list:
            dir_name = os.path.join(self.raw_data, classes)
            temp_image = self.load_images_from_directory(dir_name)
            logger.info(f"Loading images from directory - {classes}: size - {len(temp_image)}: DONE")
            image_list.extend(temp_image)

        for image in image_list:
            image.save(os.path.join(self.y, str(self.count) + '.png'))
            image = image.resize((int(image.size[0] / self.resolution), int(image.size[1] / self.resolution)))
            image.save(os.path.join(self.x, str(self.count) + '.png'))
            self.count += 1
        logger.info(f"Number of LR-HR image pairs Created: {self.count}")


if __name__ == '__main__':
    print(f'Started the Data Fetching and Processing')
    proc = process_data("/home/harsh.shukla/Caltech101/Data/101_ObjectCategories", "/home/harsh.shukla/Caltech101/Data"
                                                                                   "/SR_Images")
    print(f'Starting Image Clustering...')
    proc.process()
