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

data_len = float(len(mnist))
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

        torch.save(netD.state_dict(), "./model/weights/discriminator_{}epoch.pth".format(epoch))
        torch.save(netG.state_dict(), "./model/weights/generator_{}epoch.pth".format(epoch))

    writer.close()

def adversarial_train(aux_classifier, netG, target_classifier, hidden_dim, eps, device, epochs = 20):
    writer = SummaryWriter()
    criterion = nn.CrossEntropyLoss()
    g_optim = torch.optim.Adam( netG.parameters(), 0.0002 )
    num_of_classes = 10

    lambda1 = 0.01
    lambda2 = 100

    for epoch in range(epochs):
        epoch_l0, epoch_l1, epoch_l2 = (0, 0, 0)
        aux_acc = 0
        target_acc = 0

        for i, batch in tqdm(enumerate(train_loader)):

            # Train the discriminator and generator
            x = batch[0].to(device)
            y = batch[1].to(device)
            one_hot_y = one_hot(y).to(device)
            bsz = batch[0].shape[0]
            
            # L0 loss
            # Choose the second largest indices except for the sources
            adv_z = torch.rand((bsz, hidden_dim)).to(device)
            adv_images = netG(adv_z, one_hot_y)
            target_logits = target_classifier(adv_images)
            target_preds = F.softmax(target_logits, dim = 1)
            target_preds_idx = torch.argmax(target_preds, dim = 1)
            indices = torch.arange(10).repeat(bsz, 1).to(device)
            target_preds[indices == y.unsqueeze(1)] = 0
            adv_target_idx = torch.argmax(target_preds, dim = 1)
           
            l0 = criterion(target_logits, adv_target_idx)
                        
            # L1 loss (Check)
            l1  = torch.abs(adv_z) - eps
            l1[l1 < 0] = 0
            l1  = torch.mean(torch.sum(l1, dim = 1))

            # L2 loss
            _, aux_logits  = aux_classifier(adv_images)
            aux_preds      = F.softmax(aux_logits, dim = 1)
            aux_preds_idx = torch.argmax(aux_preds, dim = 1)
            l2  = criterion(aux_logits, y)
            
            loss = l0 + lambda1 * l1 + lambda2 * l2
            epoch_l0 += l0.item() * x.shape[0]
            epoch_l1 += l1.item() * x.shape[0]
            epoch_l2 += l2.item() * x.shape[0]
            aux_acc += sum(aux_preds_idx == y)
            target_acc += sum(target_preds_idx == y)
            
            g_optim.zero_grad()
            loss.backward()
            g_optim.step()
        
        epoch_l0 /= data_len
        epoch_l1 /= data_len
        epoch_l2 /= data_len
        aux_acc  = aux_acc.item() / data_len
        target_acc = target_acc.item() / data_len

        print("[{}/{}] L0: {:.3f}, L1: {:.3f}, L2: {:.3f}".format(epoch, epochs, epoch_l0, epoch_l1, epoch_l2))
        print("aux_accuracy: {:.3f}, target_accuracy: {:.3f}".format(aux_acc, target_acc))
        torch.save(netG.state_dict(), "./model/adv_weights/adv_generator_{}epoch.pth".format(epoch))    
            
