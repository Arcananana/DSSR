import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, inChannels=128, kSize=3):
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
    def __init__(self, inChannels=128, num_resblocks=4, kSize=3):
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
        self.conv3 = nn.Conv2d(inChannels, inChannels, 3, padding=1, stride=1)
        self.conv4 = nn.Conv2d(inChannels, inChannels, 3, padding=1, stride=1)
        self.conv_gama = nn.Conv2d(inChannels, inChannels, 1, padding=0, stride=1)
        self.conv_beta = nn.Conv2d(inChannels, inChannels, 1, padding=0, stride=1)
        self.act = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.act(self.conv1(x))
        out = self.act(self.conv2(out))
        out = self.act(self.conv3(out))
        out = self.act(self.conv4(out))
        gama = self.sigmoid(self.conv_gama(out))
        beta = self.conv_beta(out)

        return gama, beta

class DSSR(nn.Module):
    def __init__(self, color_channels=3, inChannels=128, scale=4, num_steps=4, feat_blocks=20,refine_blocks=5):
        super(DSSR, self).__init__()
        kSize = 3
        self.upscale_factor = scale
        self.num_steps = num_steps
        self.feat_blocks = feat_blocks
        self.refine_blocks = refine_blocks

        self.conv1 = nn.Conv2d(color_channels, 128, kSize, padding=(kSize-1) // 2, stride=1)
        self.feat_ext = FeatureExtractor(inChannels=128, num_resblocks=feat_blocks)
        self.conv_recon = nn.Conv2d(128, color_channels, kSize, padding=(kSize-1) // 2, stride=1)
        self.spatial_trans1 = SpatialTransformation(128)
        self.spatial_trans2 = SpatialTransformation(128)
        self.conv2 = nn.Conv2d(color_channels, 128, kSize, padding=(kSize-1) // 2, stride=1)
        self.compress_in = nn.Conv2d(2 * 128, 128, kernel_size=1, padding=0, stride=1)
        self.feat_refine = FeatureExtractor(inChannels=128, num_resblocks=refine_blocks)

        self.should_ret = True
        self.last_hidden = None

        if scale == 2:
            self.up = nn.ConvTranspose2d(inChannels, inChannels, kernel_size=4, padding=1, stride=2)
            self.down = nn.Conv2d(inChannels, inChannels, kernel_size=4, padding=1, stride=2)
        elif scale == 3:
            self.up = nn.ConvTranspose2d(inChannels, inChannels, kernel_size=5, padding=1, stride=3)
            self.down = nn.Conv2d(inChannels, inChannels, kernel_size=5, padding=1, stride=3)
        elif scale == 4:
            self.up = nn.ConvTranspose2d(inChannels, inChannels, kernel_size=4, padding=0, stride=4)
            self.down = nn.Conv2d(inChannels, inChannels, kernel_size=4, padding=0, stride=4)

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

        outs = []
        details = []
        for _ in range(self.num_steps):
            if self.should_ret:
                self.last_hidden = torch.zeros(x.size()).cuda()
                self.last_hidden.copy_(x)
                self.should_ret = False

            x = torch.cat((x, self.last_hidden), dim=1)
            x = self.compress_in(x)

            feat = self.feat_ext(x)
            up_feat = self.up(feat)
            up_recon = self.conv_recon(up_feat)
            detail = up_recon - inter_res
            details.append(detail)
            detail_feat = self.conv2(detail)

            # HR spatial transformation
            gama1, beta1 = self.spatial_trans1(detail_feat)
            up_st = up_feat * gama1 + beta1 + up_feat

            # LR spatial transformation
            detail_feat_down = F.interpolate(detail_feat, scale_factor=1. / self.upscale_factor,
                                             mode='bicubic', align_corners=True)
            down_feat = self.down(up_st)
            gama2, beta2 = self.spatial_trans2(detail_feat_down)
            down_feat = down_feat * gama2 + beta2 + feat

            self.last_hidden = down_feat

            down_feat = self.feat_refine(down_feat)
            out = self.UPNet(down_feat)
            out = out + inter_res
            outs.append(out)

        output = outs[0]
        for i in range(1, self.num_steps):
            output = output + outs[i]
        output = output / self.num_steps

        return details, outs, output


    def reset_state(self):
        self.should_ret = True