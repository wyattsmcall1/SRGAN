import argparse, time, torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage

from model import Generator

parser = argparse.ArgumentParser(description='Test Single Image')
parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
parser.add_argument('--test_mode', default='CPU', type=str, choices=['GPU', 'CPU'], help='using GPU or CPU')
parser.add_argument('--image_name', type=str, help='test low resolution image name')
# parser.add_argument('--model_name', default='netG_epoch_4_100.pth', type=str, help='generator model epoch name')
parser.add_argument('--model_name', default='netG_epoch_4_100_tom.pth', type=str, help='generator model epoch name')
opt = parser.parse_args()

U_FACTOR = opt.upscale_factor
MODE = True if opt.test_mode == 'GPU' else False
IMAGE_NAME, MODEL_NAME = opt.image_name,  opt.model_name

model = Generator(U_FACTOR).eval()

if MODE:
    model.cuda()
    # model.load_state_dict(torch.load('epochs/' + MODEL_NAME))
    model.load_state_dict(torch.load('../pth/' + MODEL_NAME))
else:
    # model.load_state_dict(torch.load('epochs/' + MODEL_NAME, map_location=lambda storage, loc: storage))
    model.load_state_dict(torch.load('../pth/' + MODEL_NAME, map_location=lambda storage, loc: storage))

image = Image.open(IMAGE_NAME)

with torch.no_grad():
    image = Variable(ToTensor()(image)).unsqueeze(0)

if MODE:
    image = image.cuda()

start = time.process_time()
out = model(image)
elapsed = (time.process_time() - start)
print('cost' + str(elapsed) + 's')
out_img = ToPILImage()(out[0].data.cpu())
# out_img.save('out_srf_' + str(U_FACTOR) + '_' + IMAGE_NAME)
out_img.save('folder_results/' + IMAGE_NAME)
