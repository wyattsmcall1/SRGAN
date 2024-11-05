from os import listdir
from os.path import join
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize

import random


possible_interpolations = [Image.BICUBIC, Image.LANCZOS, Image.HAMMING]


def display():
    return Compose([ToPILImage(), Resize(400), CenterCrop(400), ToTensor()])


def train_hr(crop_size):
    return Compose([RandomCrop(crop_size), ToTensor(), ])


def train_lr(crop_size, upscale_factor):
    chosen = random.choice(possible_interpolations)
    return Compose([ToPILImage(), Resize(crop_size // upscale_factor, interpolation=chosen), ToTensor()])


class TrainFromFolder(Dataset):
    def __init__(self, folder, crop_size, upscale_factor):
        super(TrainFromFolder, self).__init__()
        self.image_files = [join(folder, x) for x in listdir(folder) if is_image(x)]
        crop_size = valid_crop_size(crop_size, upscale_factor)
        self.hr_transform = train_hr(crop_size)
        self.lr_transform = train_lr(crop_size, upscale_factor)
        self.crop_size = crop_size
        self.upscale_factor = upscale_factor  

    def __getitem__(self, index):
        hr_image = self.hr_transform(Image.open(self.image_files[index]))
        transformer = train_lr(self.crop_size, self.upscale_factor)
        lr_image = transformer(hr_image)
        return lr_image, hr_image

    def __len__(self):
        return len(self.image_files)


class ValidateFromFolder(Dataset):
    def __init__(self, folder, upscale_factor):
        super(ValidateFromFolder, self).__init__()
        self.upscale_factor = upscale_factor
        self.image_files = [join(folder, x) for x in listdir(folder) if is_image(x)]

    def __getitem__(self, index):
        hr_image = Image.open(self.image_files[index])
        w, h = hr_image.size
        crop_size = valid_crop_size(min(w, h), self.upscale_factor)
        lr_scale = Resize(crop_size // self.upscale_factor, interpolation=random.choice(possible_interpolations))
        hr_scale = Resize(crop_size, interpolation=Image.BICUBIC)
        hr_image = CenterCrop(crop_size)(hr_image)
        lr_image = lr_scale(hr_image)
        hr_restore_img = hr_scale(lr_image)
        return ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)

    def __len__(self):
        return len(self.image_files)


class TestFromFolder(Dataset):
    def __init__(self, folder, upscale_factor):
        super(TestFromFolder, self).__init__()
        self.lr_path = folder + '/SRF_' + str(upscale_factor) + '/data/'
        self.hr_path = folder + '/SRF_' + str(upscale_factor) + '/target/'
        self.upscale_factor = upscale_factor
        self.lr_files = [join(self.lr_path, x) for x in listdir(self.lr_path) if is_image(x)]

    def __getitem__(self, index):
        image_name = self.lr_files[index].split('/')[-1]
        lr_image = Image.open(self.lr_files[index])
        w, h = lr_image.size
        hr_scale = Resize((self.upscale_factor * h, self.upscale_factor * w), interpolation=Image.BICUBIC)
        hr_restore_img = hr_scale(lr_image)
        return image_name, ToTensor()(lr_image), ToTensor()(hr_restore_img)

    def __len__(self):
        return len(self.lr_files)


def is_image(filename):
    return any(filename.endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


def valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)
