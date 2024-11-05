import argparse, os, torch
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_utils import TestFromFolder
from model import Generator

parser = argparse.ArgumentParser(description='Test Benchmark Datasets')
parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
parser.add_argument('--model_name', default='netG_epoch_4_100.pth', type=str, help='generator model epoch name')
opt = parser.parse_args()

U_FACTOR = opt.upscale_factor
MODEL_NAME = opt.model_name

model = Generator(U_FACTOR).eval()

if torch.cuda.is_available():
    model = model.cuda()
    model.load_state_dict(torch.load('epochs/' + MODEL_NAME))
else:
    model.load_state_dict(torch.load('epochs/' + MODEL_NAME, map_location=lambda storage, loc: storage))

test_set = TestFromFolder('data/test', upscale_factor=U_FACTOR)
test_loader = DataLoader(dataset=test_set, num_workers=4, batch_size=1, shuffle=False)
test_bar = tqdm(test_loader, desc='[testing from folder]')

out_path = 'from_folder_results/SRF_' + str(U_FACTOR) + '/'
if not os.path.exists(out_path):
    os.makedirs(out_path)


for image_name, lr_image, hr_restore_img in test_bar:
    image_name = image_name[0]

    with torch.no_grad():
        lr_image = Variable(lr_image)

    if torch.cuda.is_available():
        lr_image = lr_image.cuda()

    sr_image = model(lr_image)

    print(image_name.split('.')[0])
    print(image_name.split('.')[-1])
    utils.save_image(sr_image, out_path + image_name.split('.')[0] + '.' + image_name.split('.')[-1], padding=5)
