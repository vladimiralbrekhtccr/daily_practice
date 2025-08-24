import torch
from torch import nn
from torch.nn import functional as F

from decoder import VAE_AttentionBlock, VAE_ResidualBlock

class VAE_Encoder(nn.Sequential):

    def __init__(self):
        super().__init__(
            # input: (batch_size, channels=3, height=512, width=512) ->out: (batch_size, 128, height=512, width=512)
            nn.Conv2d(3, 128, kernel_size=3, padding=1),


            #(batch_size, 128, height, width) ->same_shape_out:(batch_size, 128, height, width)
            VAE_ResidualBlock(128, 128),
            # (batch_size, 128, height, width) ->same_shape_out:(batch_size, 128, height, width)
            VAE_ResidualBlock(128, 128),

            #(batch_size, 128, height, width) ->out:(batch_size, 128, height / 2 , width / 2)
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),

            # #(batch_size, 128, height / 2 , width / 2) ->out:(batch_size, 256, height / 2 , width / 2)
            VAE_ResidualBlock(128, 256),
            VAE_ResidualBlock(256, 256),

            #(batch_size, 256, height, width) ->out:(batch_size, 256, height / 4 , width / 4)
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),


            VAE_ResidualBlock(256, 512),
            VAE_ResidualBlock(512, 512),

            #(batch_size, 512, height, width) ->out:(batch_size, 512, height / 8 , width / 8)
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0), 

            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),

            # the goal of the attention block is to relate the pixels to each other.
            VAE_AttentionBlock(512),

            VAE_ResidualBlock(512, 512),

            nn.GroupNorm(32, 512),

            nn.SiLU(),

            # (batch_size, 512, height, width) ->out:(batch_size, 8, height / 8 , width / 8)
            nn.Conv2d(512, 8, kernel_size=3, padding=1),
            # (batch_size, 8, height, width) ->out:(batch_size, 8, height / 8 , width / 8)
            nn.Conv2d(8, 8, kernel_size=1, padding=0)
        )
    
    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, channels, height, width)
        # noise: (batch_size, out_channels, height / 8, width / 8)

        for module in self:
            if getattr(module, 'stride', None) == (2, 2):
                # (pad_left, pad_top, pad_right, pad_bottom)
                x = F.pad(x, (0, 1, 0, 1)) # add layer of pixels
            x = module(x)
            
        mean, log_variance = torch.chunk(x, 2, dim=1)

        log_variance = torch.clamp(log_variance, -30, 20)

        variance = log_variance.exp() # shrink negative values to small positive ones

        #TODO: stdv what is that
        stdv = variance.sqrt()

        x = mean + stdv * noise

        x *= 0.18215

        return x


            
