import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm=None):
        super(ConvBlock, self).__init__()
        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm =='batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.conv(x))
        else:
            out = self.conv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out


class DeconvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=4, stride=2, padding=1, bias=True, activation='prelu', norm=None):
        super(DeconvBlock, self).__init__()
        self.deconv = torch.nn.ConvTranspose2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.deconv(x))
        else:
            out = self.deconv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out

class ResBlock(nn.Module):
    def __init__(self, inChannels=64, kSize=3):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inChannels, inChannels, kSize, padding=(kSize-1)//2, stride=1)
        self.conv2 = nn.Conv2d(inChannels, inChannels, kSize, padding=(kSize-1)//2, stride=1)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act(out)
        out = self.conv2(out)

        return out * 0.2 + x

class CascadeResBlock(nn.Module):
    def __init__(self, inChannels=64, num_resblocks=4, kSize=3):
        super(CascadeResBlock, self).__init__()
        blocks = []
        for _ in range(num_resblocks):
            block = ResBlock(inChannels, kSize)
            blocks.append(block)
        self.crb = nn.Sequential(*blocks)

    def forward(self, x):
        out = self.crb(x)
        return out + x

class FeatureExtractor(nn.Module):
    def __init__(self, inChannels, num_resblocks, kSize=3):
        super(FeatureExtractor, self).__init__()
        self.crb = CascadeResBlock(inChannels, num_resblocks, kSize)

    def forward(self, x):
        return self.crb(x)

class SpatialTransformation(nn.Module):
    def __init__(self, inChannels):
        super(SpatialTransformation, self).__init__()
        self.conv1 = nn.Conv2d(inChannels, inChannels, 3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(inChannels, inChannels, 3, padding=1, stride=1)
        self.conv_gama = nn.Conv2d(inChannels, inChannels, 1, padding=0, stride=1)
        self.conv_beta = nn.Conv2d(inChannels, inChannels, 1, padding=0, stride=1)
        self.feat_ext = FeatureExtractor(inChannels, num_resblocks=4, kSize=3)
        self.act = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.act(self.conv1(x))
        out = self.act(self.conv2(out))
        out=self.act(self.feat_ext(out))
        gama = self.sigmoid(self.conv_gama(out))
        beta = self.conv_beta(out)

        return gama, beta

# num_steps = 4, 8, 1
# num_resblocks = 4, 2, 16
# inChannels = 64
# color_channels = 3
# scale = 3
# loss = l1(sr_out, gt）+ l1(detail, (gt-lr_bic))

class RDN(nn.Module):
    def __init__(self, color_channels, inChannels, scale, num_steps, num_resblocks):
        super(RDN, self).__init__()
        kSize = 3
        self.upscale_factor = scale
        self.num_steps = num_steps
        self.num_resblocks = num_resblocks

        if scale == 2:
            kernel_size = 6
            stride = 2
            padding = 2
        elif scale==3:
            kernel_size=7
            stride=3
            padding=2
        elif scale == 4:
            kernel_size = 8
            stride = 4
            padding = 2
        elif scale == 8:
            kernel_size = 12
            stride = 8
            padding = 2

        num_filter=inChannels
        self.conv1 = nn.Conv2d(color_channels, inChannels, kSize, padding=(kSize-1)//2, stride=1)
        self.feat_ext = FeatureExtractor(inChannels, num_resblocks=self.num_resblocks, kSize=kSize)
        self.conv_recon = nn.Conv2d(inChannels, color_channels, kSize, padding=(kSize-1)//2, stride=1)
        self.spatial_trans1 = SpatialTransformation(inChannels)
        self.spatial_trans2 = SpatialTransformation(inChannels)
        self.conv2 = nn.Conv2d(color_channels, inChannels, kSize, padding=(kSize-1)//2, stride=1)

        self.up1=DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation='prelu', norm=None)
        self.up2=DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation='prelu', norm=None)
        self.up3=DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation='prelu', norm=None)

        self.down1 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation='prelu', norm=None)
        self.down2 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation='prelu', norm=None)
        self.down3 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation='prelu', norm=None)
        self.down4 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation='prelu', norm=None)

        self.compress_in = nn.Conv2d(2 * inChannels, inChannels, kernel_size=1, padding=0, stride=1)
        self.should_ret = True
        self.last_hidden = None


        if scale == 2 or scale == 3:
            self.UPNet = nn.Sequential(*[
                nn.Conv2d(inChannels, inChannels * scale * scale, kSize, padding=(kSize - 1) // 2, stride=1),
                nn.PixelShuffle(scale),
                nn.Conv2d(inChannels, color_channels, kSize, padding=(kSize - 1) // 2, stride=1)
            ])
        elif scale == 4:
            self.UPNet = nn.Sequential(*[
                nn.Conv2d(inChannels, inChannels * 4, kSize, padding=(kSize - 1) // 2, stride=1),
                nn.PixelShuffle(2),
                nn.Conv2d(inChannels, inChannels * 4, kSize, padding=(kSize - 1) // 2, stride=1),
                nn.PixelShuffle(2),
                nn.Conv2d(inChannels, color_channels, kSize, padding=(kSize - 1) // 2, stride=1)
            ])
        else:
            raise ValueError("scale must be 2 or 3 or 4.")

    def forward(self, x):
        self.reset_state()
        inter_res = F.interpolate(x, scale_factor=self.upscale_factor, mode='bicubic', align_corners=False)
        x = self.conv1(x)

        details=[]
        outs=[]
        for _ in range(self.num_steps):
            if self.should_ret:
                self.last_hidden=torch.zeros(x.size()).cuda()
                self.last_hidden.copy_(x)
                self.should_ret=False

            x=torch.cat((x,self.last_hidden),dim=1)
            x=self.compress_in(x)

            feat = self.feat_ext(x)

            #calculate hr feat
            up_feat = self.up1(feat)
            up_recon = self.conv_recon(up_feat)
            detail = up_recon - inter_res
            details.append(detail)
            detail_feat = self.conv2(detail)
            gama1,beta1=self.spatial_trans1(detail_feat)
            l0=self.down1(up_feat * gama1 + beta1 + up_feat)
            h1=self.up2(l0-feat)
            feat_hr=h1+up_feat

            #calculate lr feat
            detail_feat_down=self.down2(detail_feat)
            down_feat=self.down3(feat_hr)
            gama2, beta2 = self.spatial_trans2(detail_feat_down)
            l1=self.up3(down_feat*gama2+beta2+down_feat)
            h2=self.down4(l1-feat_hr)
            feat_lr=h2+down_feat

            self.last_hidden=feat_lr
            out=self.UPNet(feat_lr)
            out=out+inter_res
            outs.append(out)

        output=outs[0]
        for i in range(1,self.num_steps):
            output=output+outs[i]
        output=output/self.num_steps

        return details,outs,output


    def reset_state(self):
        self.should_ret = True



