from .baseline_fcn import FCN8s
from .baseline_unet import UNet
from .baseline_att_unet import AttentionUNet
from .baseline_deeplabv3p import DeepLabV3Plus
from .baseline_transunet import TransUNet
from .effibound_net import EffiBoundNet

def get_model(model_name, in_channels=3, num_classes=1, img_size=(256, 256)):
    name = model_name.lower().replace("_", "").replace("-", "")
    if name in ["fcn", "fcn8s"]:
        return FCN8s(in_channels=in_channels, num_classes=num_classes)
    elif name in ["unet"]:
        return UNet(in_channels=in_channels, num_classes=num_classes)
    elif name in ["attunet", "attentionunet"]:
        return AttentionUNet(in_channels=in_channels, num_classes=num_classes)
    elif name in ["deeplabv3plus", "deeplabv3p", "deeplabv3"]:
        return DeepLabV3Plus(in_channels=in_channels, num_classes=num_classes)
    elif name in ["transunet"]:
        return TransUNet(in_channels=in_channels, num_classes=num_classes, img_size=img_size)
    elif name in ["effiboundnet", "effibound", "ours"]:
        return EffiBoundNet(in_channels=in_channels, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
