from torch import nn
import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image

class VGGIntermediate(nn.Module):
    def __init__(self, layers_to_extract):
        super(VGGIntermediate, self).__init__()
        self.layers = nn.ModuleList()

        vgg_model = models.vgg16(pretrained=True)

        self.layers_to_extract = layers_to_extract
        max_layer = max(self.layers_to_extract)
        for idx, layer in enumerate(vgg_model.features):
            self.layers.append(layer)
            if idx == max_layer:
                break
        
        self.preprocess = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def forward(self, img):
        # img: shall be in [0, 1] range and in layout [B, C, H, W]
        x = self.preprocess(img)  # Apply transforms
        
        outputs = []
        
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            if idx in self.layers_to_extract:
                outputs.append(x)
        return outputs


if __name__ == "__main__":
    vgg_layers_to_extract = [4, 9, 16, 23]
    
    custom_vgg = VGGIntermediate(vgg_layers_to_extract)
    custom_vgg.eval()

    for param in custom_vgg.parameters():
        param.requires_grad = False
        
    img = torch.randint(0, 256, (1, 3, 256, 256), dtype=torch.float32)    
    outputs = custom_vgg(img)
    for layer_idx, output in zip(vgg_layers_to_extract, outputs):
        print(f"VGG Layer idx: {layer_idx}, Output shape: {output.shape}")
        # VGG Layer idx: 4, Output shape: torch.Size([1, 64, 128, 128])
        # VGG Layer idx: 9, Output shape: torch.Size([1, 128, 64, 64])
        # VGG Layer idx: 16, Output shape: torch.Size([1, 256, 32, 32])
        # VGG Layer idx: 23, Output shape: torch.Size([1, 512, 16, 16])
