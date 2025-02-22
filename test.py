import torch
import torch.nn as nn
from wtconv import WTConv2d
from thop import profile

#DPblock
class Depthwise_Separable_Conv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Depthwise_Separable_Conv, self).__init__()
        self.ch_in = in_channels
        self.ch_out = out_channels
        # self.depth_conv = WTConv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride,
        #                            wt_levels=3)  # WTConv with 3x3 kernel and 3 levels
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        self.point_conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.point_conv(x)
        return x

#WTBlock
def Depthwise_Separable_Conv2(in_channels, out_channels, kernel_size, stride):
    return nn.Sequential(
        WTConv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, wt_levels=3),
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
    )

#C+DP*3
def get_pdn_small(out_channels=384, padding=False):
    pad_mult = 1 if padding else 0
    return nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=128, kernel_size=4,
                  padding=0 * pad_mult),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),
        Depthwise_Separable_Conv(in_channels=128, out_channels=256, kernel_size=4, stride=1, padding='same'),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),
        Depthwise_Separable_Conv(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding='same'),
        nn.ReLU(inplace=True),
        Depthwise_Separable_Conv(in_channels=256, out_channels=out_channels, kernel_size=4, stride=1,
                                 padding='same'),
    )

#c+WT*3/WT*4
def gget_pdn_small(out_channels=384, padding=False):
    pad_mult = 1 if padding else 0
    return nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=128, kernel_size=4,
                  padding=0 * pad_mult),
        # Depthwise_Separable_Conv2(in_channels=3, out_channels=128, kernel_size=4, stride=1),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),
        Depthwise_Separable_Conv2(in_channels=128, out_channels=256, kernel_size=4, stride=1),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),
        Depthwise_Separable_Conv2(in_channels=256, out_channels=256, kernel_size=3, stride=1),
        nn.ReLU(inplace=True),
        Depthwise_Separable_Conv2(in_channels=256, out_channels=out_channels, kernel_size=4, stride=1),
    )


def Get_pdn_small(out_channels=384, padding=False):
    pad_mult = 1 if padding else 0
    return nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=128, kernel_size=4,
                  padding=3 * pad_mult),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4,
                  padding=3 * pad_mult),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3,
                  padding=1 * pad_mult),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=256, out_channels=out_channels, kernel_size=4)
    )

#c+DP*5
def Get_pdn_mediumv2(out_channels=384, padding=False):
    pad_mult = 1 if padding else 0
    return nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=256, kernel_size=4, stride=1,
                  padding=0 * pad_mult),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),
        Depthwise_Separable_Conv(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding='same'),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),
        Depthwise_Separable_Conv(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding='same'),
        nn.ReLU(inplace=True),
        Depthwise_Separable_Conv(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding='same'),
        nn.ReLU(inplace=True),
        Depthwise_Separable_Conv(in_channels=512, out_channels=out_channels, kernel_size=4, stride=1,
                                 padding='same'),
        nn.ReLU(inplace=True),
        Depthwise_Separable_Conv(in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=1,
                                 padding='same'),
    )

#c+WT*5/ALL WT
def get_pdn_medium(out_channels=384, padding=False):
    pad_mult = 1 if padding else 0
    return nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=256, kernel_size=4, stride=1,
                  padding=3 * pad_mult),
        # Depthwise_Separable_Conv2(in_channels=3, out_channels=256, kernel_size=4, stride=1),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),
        Depthwise_Separable_Conv2(in_channels=256, out_channels=512, kernel_size=4, stride=1),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),
        Depthwise_Separable_Conv2(in_channels=512, out_channels=512, kernel_size=1, stride=1),
        nn.ReLU(inplace=True),
        Depthwise_Separable_Conv2(in_channels=512, out_channels=512, kernel_size=3, stride=1),
        nn.ReLU(inplace=True),
        Depthwise_Separable_Conv2(in_channels=512, out_channels=out_channels, kernel_size=4, stride=1),
        nn.ReLU(inplace=True),
        Depthwise_Separable_Conv2(in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=1),
    )


def Get_pdn_medium(out_channels=384, padding=False):
    pad_mult = 1 if padding else 0
    return nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=256, kernel_size=4,
                  padding=3 * pad_mult),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),
        nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4,
                  padding=3 * pad_mult),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,
                  padding=1 * pad_mult),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=512, out_channels=out_channels, kernel_size=4),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                  kernel_size=1)
    )


def Get_ae():
    return nn.Sequential(
        # encoder
        nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(inplace=True),
        nn.Conv2d(32, 32, 4, 2, 1), nn.ReLU(inplace=True),
        nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, 4, 2, 1), nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, 4, 2, 1), nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, 8),
        # decoder
        nn.Upsample(3, mode='bilinear'),
        nn.Conv2d(64, 64, 4, 1, 2), nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(8, mode='bilinear'),
        nn.Conv2d(64, 64, 4, 1, 2), nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(15, mode='bilinear'),
        nn.Conv2d(64, 64, 4, 1, 2), nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(32, mode='bilinear'),
        nn.Conv2d(64, 64, 4, 1, 2), nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(63, mode='bilinear'),
        nn.Conv2d(64, 64, 4, 1, 2), nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(127, mode='bilinear'),
        nn.Conv2d(64, 64, 4, 1, 2), nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(64, mode='bilinear'),
        nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(inplace=True),
        nn.Conv2d(64, 384, 3, 1, 1)
    )

def get_autoencoder(out_channels=384):
    return nn.Sequential(
        # encoder
        nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4, stride=2,
                  padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2,
                  padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2,
                  padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2,
                  padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2,
                  padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=8),
        # decoder
        nn.Upsample(size=3, mode='bilinear'),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
                  padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=8, mode='bilinear'),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
                  padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=15, mode='bilinear'),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
                  padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=32, mode='bilinear'),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
                  padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=63, mode='bilinear'),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
                  padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=127, mode='bilinear'),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
                  padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=56, mode='bilinear'),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1,
                  padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=3,
                  stride=1, padding=1)
    )


def print_model_flops_and_params(teacher, student, autoencoder, input_tensor):
    # 获取 teacher 模型的参数和FLOPs
    flops_teacher, params_teacher = profile(teacher, inputs=(input_tensor,))
    flops_teacher /= 10 ** 9  # 转换为 10^9
    params_teacher /= 10 ** 6  # 转换为 10^6
    print(f"Teacher Model - FLOPs: {flops_teacher:.2f}G, Params: {params_teacher:.2f}M")

    # 获取 student 模型的参数和FLOPs
    flops_student, params_student = profile(student, inputs=(input_tensor,))
    flops_student /= 10 ** 9  # 转换为 10^9
    params_student /= 10 ** 6  # 转换为 10^6
    print(f"Student Model - FLOPs: {flops_student:.2f}G, Params: {params_student:.2f}M")

    # 获取 autoencoder 模型的参数和FLOPs
    flops_autoencoder, params_autoencoder = profile(autoencoder, inputs=(input_tensor,))
    flops_autoencoder /= 10 ** 9  # 转换为 10^9
    params_autoencoder /= 10 ** 6  # 转换为 10^6
    print(f"Autoencoder Model - FLOPs: {flops_autoencoder:.2f}G, Params: {params_autoencoder:.2f}M")
    print(
        f"Total - FLOPs: {flops_autoencoder + flops_student + flops_teacher:.2f}G, Params: {params_autoencoder + params_student + params_teacher:.2f}M")


if __name__ == "__main__":
    from torchsummary import summary

    #
    dummy_input = torch.randn(2, 3, 256, 256).cuda()
    # mod1 = Get_pdn_mediumv2(384, True).cuda()
    # pdnt = Get_pdn_medium(384).cuda()
    # pdns = Get_pdn_medium(768).cuda()
    # PDNAE = Get_ae().cuda()
    # # wcPDN
    # teacher = get_pdn_medium(384).cuda()
    # student = get_pdn_medium(768).cuda()
    ae = get_autoencoder().cuda()
    # aee = GGet_ae().cuda()
    # # m = get_ae().cuda()
    # # mm = Get_ae().cuda()
    # spdnt = gget_pdn_small(384).cuda()
    # spdntT = gget_pdn_small(768).cuda()
    # #
    # summary(pdnt, (3, 256, 256))
    # summary(spdnt, (3, 256, 256))
    # # summary(pdnt, (3, 256, 256))
    # # summary(pdns, (3, 256, 256))
    # # # summary(student, (3, 256, 256))
    # # #
    # spdn = Get_pdn_small(384).cuda()
    # spdns = Get_pdn_small(768).cuda()
    # summary(spdn, (3, 256, 256))
    # t = get_pdn_small(384).cuda()
    # s = get_pdn_small(768).cuda()
    # # our
    # print_model_flops_and_params(teacher, student, PDNAE, dummy_input)
    # # pdn
    # print_model_flops_and_params(pdnt, pdns, PDNAE, dummy_input)
    # print_model_flops_and_params(spdnt, spdns, PDNAE, dummy_input)
    # print_model_flops_and_params(t, s, PDNAE, dummy_input)
    T = get_pdn_medium(384).cuda()
    s = get_pdn_medium(768).cuda()
    summary(T, (3, 256, 256))
    # # our
    # # # our
    # print_model_flops_and_params(spdnt, spdntT, aee, dummy_input)
    # print_model_flops_and_params(spdn, spdns, ae, dummy_input)
    # print_model_flops_and_params(pdnt, pdns, ae, dummy_input)
    print_model_flops_and_params(T, s, ae, dummy_input)
