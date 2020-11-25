import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, ):
        self.res1 = ResidualBlockDisc()
        self.res2 = ResidualBlock()
        self.res3 = ResidualBlock()
        self.res4 = ResidualBlock()
        

    def forward(self, x):

class Generator(nn.Module):
    def __init__(self, input_size):
        self.preprocess  = nn.Linear()
        self.resblocks1  = DenseSequential([
                                            Dense(nn.Conv2d(16,32,)),
                                            nn.ReLU(),
                                            Dense(nn.Conv2d(16,32,), None, 
                                                  nn.Conv2d(32,32,))
                                            ])

        self.resblocks2  = DenseSequential([
                                            Dense(nn.Conv2d(16,32,)),
                                            nn.ReLU(),
                                            Dense(nn.Conv2d(16,32,), None, 
                                                  nn.Conv2d(32,32,))
                                            ])

        self.final_layer = nn.Sequential(
                                          nn.BatchNorm2d(input_size),
                                          nn.ReLU(),
                                          nn.Conv2d(),
                                          nn.Sigmoid()
                                        )
    def forward(self, z, y):
        onehot   = label_to_onehot(y)
        z_onehot = torch.cat([z,y], dim = 1) 
        out = self.preprocess(z_onehot)
        out = self.resblocks1(out)
        out = self.resblocks2(out)
        out = self.final_layer(out)


class ACWGAN(nn.Module):
    def __init__(self, dim_z, dim_G = 128, dim_D = 128):
        self.discriminator = Discriminator(dim_D)
        self.generator     = Generator(dim_G)
