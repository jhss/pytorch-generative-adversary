import torch
import torch.nn as nn
import torchvision
import torch.utils.data as data
import torch.nn.functional as F

from torchvision import transforms
from model.acwgan_temp import FirstResBlockDiscriminator, ResBlockDiscriminator
from tqdm import tqdm

channels = 1
DISC_SIZE = 64

class Classifier(nn.Module):
    def __init__(self, num_of_classes):
        super(Classifier, self).__init__()
        self.encoder = nn.Sequential(
                FirstResBlockDiscriminator(channels, DISC_SIZE, stride = 2),
                ResBlockDiscriminator(DISC_SIZE, DISC_SIZE * 2, stride = 2),
                ResBlockDiscriminator(DISC_SIZE*2, DISC_SIZE*4, stride = 2),
                ResBlockDiscriminator(DISC_SIZE*4, DISC_SIZE*8, stride = 2),
                nn.ReLU(),
             )
        self.classifier = nn.Linear(DISC_SIZE*8, num_of_classes)
        nn.init.xavier_uniform_(self.classifier.weight.data, 1)

    def forward(self, x):
        h = self.encoder(x).view(-1, DISC_SIZE * 8)
        return self.classifier(h)

if __name__ == '__main__':

    transform = transforms.Compose([ transforms.ToTensor(),
                                     transforms.Normalize(mean = [0.5], std = [0.5]),
                                   ])
    test_transform = transforms.Compose([ transforms.ToTensor() ])
    train_mnist = torchvision.datasets.MNIST(root = "/data1/home/juhong/research/datasets/mnist",
                                             train = True, download = True, transform = transform)
    test_mnist  = torchvision.datasets.MNIST(root = "/data1/home/juhong/research/datasets/mnist",
                                             train = True, download = True, transform = transform)
    
    epochs, bsz = (30, 32)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader = data.DataLoader(train_mnist, batch_size = bsz, shuffle = True)
    test_loader  = data.DataLoader(test_mnist, batch_size = bsz, shuffle = True)
    netC = Classifier(10).to(device)

    optim = torch.optim.Adam(netC.parameters(), weight_decay = 1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        epoch_loss = 0
        acc = 0
        for i, batch in tqdm(enumerate(train_loader)):
            x = batch[0].to(device)
            y = batch[1].to(device)

            preds = netC(x)
            loss  = criterion(preds, y)
            preds_idx = torch.argmax(F.softmax(preds, dim = 1), dim = 1)

            optim.zero_grad()
            loss.backward()
            optim.step()

            epoch_loss += loss.item() * x.shape[0]
            acc += sum(preds_idx == y)
        
        epoch_loss /= float(len(train_mnist))
        
        print("epoch_loss: ", epoch_loss)
        print("Accuracy: ", acc / float(len(train_mnist)))
        torch.save(netC.state_dict(), "/data1/home/juhong/research/adversarial/pytorch-generative-adversary/model/weights/classifier_epoch{}.pth".format(epoch))


