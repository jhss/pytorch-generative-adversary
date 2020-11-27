import torch
import torch.nn as nn
import torchvision
import argparse
import os

from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from model.acwgan_temp import *
from model.classifier import *
from util import *
from tqdm import tqdm
from trainer import *

parser = argparse.ArgumentParser("Latent Provable Defense Network")
parser.add_argument("--mode", type = str, default = 'standard', help = 'standard | adversarial | attack')
num_of_classes = 10
hidden_dim = 512
eps = 8/255
load_epoch = 13

base_path = "/data1/home/juhong/research/adversarial/pytorch-generative-adversary"
generator_path = os.path.join(base_path, "model/weights/generator_{}epoch.pth".format(load_epoch))
aux_classifier_path = os.path.join(base_path, "model/weights/discriminator_{}epoch.pth".format(load_epoch))
target_classifier_path = os.path.join(base_path, "model/weights/classifier_epoch{}.pth".format(load_epoch))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
netD = Discriminator(c_dim = num_of_classes).to(device)
netG = Generator(z_dim = hidden_dim, c_dim = num_of_classes).to(device)
target_classifier = Classifier(num_of_classes).to(device)

args = parser.parse_args()

if args.mode == 'standard':
    gan_train(netD, netG, device)
elif args.mode == 'adversarial':
    netG.load_state_dict(torch.load(generator_path))
    netD.load_state_dict(torch.load(aux_classifier_path))
    target_classifier.load_state_dict(torch.load(target_classifier_path))
    adversarial_train(netD, netG, target_classifier, hidden_dim, eps, device, epochs=20)
elif args.mode == 'attack':
    writer = SummaryWriter(log_dir = "./adv_runs/")
    
    load_epoch = 0
    adv_generator_path = os.path.join(base_path, 
                                        "model/adv_weights/adv_generator_{}epoch.pth".format(load_epoch))
    netG.load_state_dict(torch.load(adv_generator_path))
    netD.load_state_dict(torch.load(aux_classifier_path))
    target_classifier.load_state_dict(torch.load(target_classifier_path))
    
    target_classifier.eval()
    netD.eval()
    netG.eval()

    adv_z = torch.randn(32, hidden_dim).to(device)
    y = torch.arange(10).repeat(3)
    y = torch.cat([y, torch.tensor([0,1])], dim = 0)
    one_hot_y = one_hot(y).to(device)

    adv_images = netG(adv_z, one_hot_y)
    
    

    
    grid = torchvision.utils.make_grid(adv_images)
    writer.add_image("adversarial_images_epoch{}".format(load_epoch), grid, 0)
    writer.close()

