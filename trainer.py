import torch
import torch.nn as nn
import torchvision

from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from model.acwgan_temp import *
from util import *
from tqdm import tqdm

transform = transforms.Compose([ transforms.ToTensor(),
                                 transforms.Normalize( mean = [0.5], std = [0.5] )
                               ])
mnist = torchvision.datasets.MNIST("/data1/home/juhong/research/datasets/mnist", 
                                   transform = transform)

train_loader = torch.utils.data.DataLoader(dataset = mnist, batch_size = 32, shuffle = True)

def gan_train(netD, netG, device, lamb = 10, epochs = 20, hidden_dim = 512):

    writer = SummaryWriter()
    criterion = nn.CrossEntropyLoss()
    d_optim = torch.optim.Adam( netD.parameters(), 0.0002 )
    g_optim = torch.optim.Adam( netG.parameters(), 0.0002 )
    num_of_classes = 10

    for epoch in range(epochs):
        for i, batch in tqdm(enumerate(train_loader)):

            # Train the discriminator and generator

            x = batch[0].to(device)
            y = batch[1].to(device)
            one_hot_y = one_hot(y).to(device)
            z = torch.randn(x.shape[0], hidden_dim).to(device)

            fake_images = netG(z, one_hot_y)
            r_tf_logit, r_class_logits = netD(x)
            f_tf_logit, f_class_logits = netD(fake_images)

            r_tf_loss = torch.mean(r_tf_logit, dim = 0)
            f_tf_loss = torch.mean(f_tf_logit, dim = 0)
            r_ce = torch.mean(criterion(r_class_logits, y), dim = 0)

            # Discriminator and Generator Loss
            d_loss = f_tf_loss - r_tf_loss + r_ce

            # Optimize the discriminator with the discriminator loss
            d_optim.zero_grad()
            d_loss.backward()
            d_optim.step()

            # Gradient Penalty
            alpha = torch.rand(x.shape[0], 1, 1, 1).to(device).expand_as(x)
            interpolates = Variable(alpha * x + (1 - alpha) * fake_images, requires_grad = True)
            inter_out, _ = netD(interpolates)

            grad = torch.autograd.grad(outputs = inter_out, inputs = interpolates,
                                       grad_outputs = torch.ones(inter_out.size()).to(device),
                                       retain_graph = True, create_graph = True, only_inputs = True)[0]
            grad = grad.view(grad.size(0), -1)
            grad_l2_norm = torch.sqrt(torch.sum(grad ** 2, dim = 1))
            gradient_penalty = torch.mean((grad_l2_norm - 1) ** 2)

            d_loss = lamb * gradient_penalty
            
            # Optimize the discriminator with the gradient penalty
            d_optim.zero_grad()
            d_loss.backward()
            d_optim.step()

            # Optimizer the generator with the generator loss
            fake_images = netG(z, one_hot_y)
            f_tf_logit, f_class_logits = netD(fake_images)
            f_tf_loss = torch.mean(f_tf_logit, dim = 0)
            f_ce = torch.mean(criterion(f_class_logits, y), dim = 0)

            g_loss = -f_tf_loss + f_ce
            g_optim.zero_grad()
            g_loss.backward()
            g_optim.step()
            
        
        print("{} epoch".format(epoch))
        grid  = torchvision.utils.make_grid(fake_images)
        grid2 = torchvision.utils.make_grid(x) 
        writer.add_image("fake_images_epoch{}".format(epoch), grid, 0)
        writer.add_image("original_images_epoch{}".format(epoch), grid2, 0)

    writer.close()
