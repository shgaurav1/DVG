import torch
import torch.nn as nn
import torchvision

class MaskL1Loss(nn.Module):
    """
    Loss from paper <Pose Guided Person Image Generation> Sec3.1 pose mask loss
    """

    def __init__(self, ratio=3):
        super(MaskL1Loss, self).__init__()
        self.criterion = nn.L1Loss()
        self.ratio = ratio

    def forward(self, generated_img, target_img, mask):
        pose_mask_l1 = self.criterion(generated_img * mask, target_img * mask)
        return pose_mask_l1 * self.ratio #self.criterion(generated_img, target_img) +



class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))
        self.resize = resize

    def forward(self, input, target):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += torch.nn.functional.l1_loss(x, y)
        return loss
        

class TotalLoss(torch.nn.Module):
    def __init__(self):
        super(TotalLoss,self).__init__()
        self.L1 = nn.L1Loss()
        self.Mask = MaskL1Loss() 
        self.perceptualLoss = VGGPerceptualLoss()

    def forward(self,input,target):#,mask):
        L1_loss = self.L1(input,target)
        # Mask_loss = self.Mask(input,target,mask)
        perceptual_total = self.perceptualLoss(input,target)
        # mask_perceptual = self.perceptualLoss(input*mask, target*mask)

        return L1_loss+ perceptual_total 
