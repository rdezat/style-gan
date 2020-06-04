import torch
import torchvision

class GeneratorNet(torch.nn.Module):
    def __init__(self):
        super(GeneratorNet, self).__init__()
        # # Initial convolution layers
        # self.conv1 = ConvLayer(3, 32, kernel_size=9, stride=1)
        # self.in1 = torch.nn.InstanceNorm2d(32, affine=True)
        # self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        # self.in2 = torch.nn.InstanceNorm2d(64, affine=True)
        # self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        # self.in3 = torch.nn.InstanceNorm2d(128, affine=True)
        # # Residual layers
        # self.res1 = ResidualBlock(128)
        # self.res2 = ResidualBlock(128)
        # self.res3 = ResidualBlock(128)
        # self.res4 = ResidualBlock(128)
        # self.res5 = ResidualBlock(128)
        # # Upsampling Layers
        # self.deconv1 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)
        # self.in4 = torch.nn.InstanceNorm2d(64, affine=True)
        # self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        # self.in5 = torch.nn.InstanceNorm2d(32, affine=True)
        # self.deconv3 = ConvLayer(32, 3, kernel_size=9, stride=1)
        # # Non-linearities
        # self.relu = torch.nn.ReLU()
     
        # Initial lum convolution layers
        self.conv1_lum = ConvLayer(1, 28, kernel_size=9, stride=2)
        self.in1 = torch.nn.InstanceNorm2d(28, affine=True)
        self.conv2 = ConvLayer(28, 50, kernel_size=3, stride=2)
        self.in2 = torch.nn.InstanceNorm2d(50, affine=True)
        self.conv3 = ConvLayer(50, 50, kernel_size=3, stride=1)
        self.in3 = torch.nn.InstanceNorm2d(50, affine=True)
         # Initial chroma convolution layers
        self.conv1_chr = ConvLayer(2, 28, kernel_size=9, stride=1)
        # Residual layers
        self.res1 = ResidualBlock(50)
        self.res2 = ResidualBlock(50)
        self.res3 = ResidualBlock(50)
        self.res4 = ResidualBlock(50)
        self.res5 = ResidualBlock(50)
        # Upsampling lum Layers
        self.deconv1 = UpsampleConvLayer(50, 28, kernel_size=3, stride=1, upsample=2)
        self.in4 = torch.nn.InstanceNorm2d(28, affine=True)
        self.deconv2 = UpsampleConvLayer(28, 1, kernel_size=3, stride=1, upsample=2)
        self.in5 = torch.nn.InstanceNorm2d(1, affine=True)
        self.deconv3_lum = ConvLayer(1, 1, kernel_size=9, stride=1)
        # Upsampling chroma Layers
        self.deconv2_chr = UpsampleConvLayer(28, 28, kernel_size=3, stride=1, upsample=1)
        self.in5_chr = torch.nn.InstanceNorm2d(28, affine=True)
        self.deconv3_chr = UpsampleConvLayer(28, 2, kernel_size=3, stride=1, upsample=1)
        # Non-linearities
        self.relu = torch.nn.ReLU()

    def forward(self, X):
        # y = self.relu(self.in1(self.conv1(X)))
        # y = self.relu(self.in2(self.conv2(y)))
        # y = self.relu(self.in3(self.conv3(y)))
        # y = self.res1(y)
        # y = self.res2(y)
        # y = self.res3(y)
        # y = self.res4(y)
        # y = self.res5(y)
        # y = self.relu(self.in4(self.deconv1(y)))
        # y = self.relu(self.in5(self.deconv2(y)))
        # y = self.deconv3(y)
        # Luminance
        # print(X)
        y_lum = self.relu(self.in1(self.conv1_lum(X[:,0,:,:].unsqueeze(1))))
        y_lum = self.relu(self.in2(self.conv2(y_lum)))
        y_lum = self.relu(self.in3(self.conv3(y_lum)))
        y_lum = self.res1(y_lum)
        y_lum = self.res2(y_lum)
        y_lum = self.res3(y_lum)
        y_lum = self.res4(y_lum)
        y_lum = self.res5(y_lum)
        y_lum = self.relu(self.in4(self.deconv1(y_lum)))
        y_lum = self.relu(self.in5(self.deconv2(y_lum)))
        y_lum = self.deconv3_lum(y_lum)
        # Chroma
        y_chr = self.relu(self.in1(self.conv1_chr(X[:,1:,:,:])))
        y_chr = self.relu(self.in2(self.conv2(y_chr)))
        y_chr = self.relu(self.in3(self.conv3(y_chr)))
        y_chr = self.res1(y_chr)
        y_chr = self.res2(y_chr)
        y_chr = self.res3(y_chr)
        y_chr = self.res4(y_chr)
        y_chr = self.res5(y_chr)
        y_chr = self.relu(self.in4(self.deconv1(y_chr)))
        y_chr = self.relu(self.in5_chr(self.deconv2_chr(y_chr)))
        y_chr = self.deconv3_chr(y_chr)
        y = torch.zeros(X.shape, dtype=torch.float32)
        y[:,0,:,:] = y_lum[:,0,:,:]
        y[:,1:,:,:] = y_chr
        # print(y)
        return y

    
class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class ResidualBlock(torch.nn.Module):
    """ResidualBlock
    introduced in: https://arxiv.org/abs/1512.03385
    recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html
    """

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        return out


class UpsampleConvLayer(torch.nn.Module):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = torch.nn.functional.interpolate(x_in, mode='nearest', scale_factor=self.upsample)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out
