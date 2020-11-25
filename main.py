import torch
import torch.nn as nn
import torchvision
import argparse

from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from model.acwgan_temp import *
from util import *
from tqdm import tqdm
from trainer import *

parser = argparse.ArgumentParser("Latent Provable Defense Network")
parser.add_argument("--mode", type = str, default = 'standard', help = 'standard | adversarial | attack')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
netD = Discriminator(c_dim = num_of_classes).to(device)
netG = Generator(z_dim = hidden_dim, c_dim = num_of_classes).to(device)

args = parser.parse_args()

if args.mode == 'standard':
    gan_train(netD, neG, device)
elif args.mode == 'adversarial':
    adversarial_train(netD, netG, aux_classifier, target_classifier)
