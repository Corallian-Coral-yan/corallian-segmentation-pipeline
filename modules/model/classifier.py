import torch
import torch.nn as nn
from modules.model.resnet_hdc import ResNet18_HDC, ResNet101_HDC
from modules.model.aspp import ASPP


class DUC(nn.Module):
    """Dense Upsampling Convolution (DUC) for learned upsampling."""

    def __init__(self, in_channels, up_factor):
        super(DUC, self).__init__()
        self.duc = nn.Conv2d(in_channels, in_channels *
                             up_factor**2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_factor)  # Upsamples spatially

    def forward(self, x):
        x = self.duc(x)
        x = self.pixel_shuffle(x)
        return x


class CoralSegmentationModel(nn.Module):
    """
    Coral segmentation pipeline:
    Input -> ResNet with HDC -> ASPP -> DUC -> 1x1 Convolution -> Segmentation Mask
    """

    def __init__(self, config):
        super(CoralSegmentationModel, self).__init__()

        self.config = config
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # Select ResNet backbone
        if self.config["resnet"]["ResNetModel"] == 18:
            resnet_model = ResNet18_HDC
        elif self.config["resnet"]["ResNetModel"] == 101:
            resnet_model = ResNet101_HDC
        else:
            raise ValueError("Invalid ResNet Configuration (Use 18 or 101)")

        # Initialize ResNet model
        self.feature_extractor = resnet_model().to(self.device)

        # Load cached model if specified
        if self.config["UseCachedModel"]:
            model_path = self.config["ModelFilepath"]
            print(f"Using cached model at {model_path}")
            self.feature_extractor.load_state_dict(
                torch.load(model_path, map_location=self.device))

        # Extract feature maps from ResNet (excluding classification layers)
        self.feature_extractor = nn.Sequential(
            *list(self.feature_extractor.children())[:-2])

        # ASPP Configuration
        aspp_config = self.config["aspp"]
        self.aspp_enabled = aspp_config.get("ASPPEnabled", True)
        self.aspp_in_channels = aspp_config.get("ASPPInChannels", 1024)
        self.aspp_out_channels = aspp_config.get("ASPPOutChannels", 256)
        self.atrous_rates = aspp_config.get("AtrousRates", [6, 12, 18])

        # ASPP module (only if enabled)
        if self.aspp_enabled:
            print(
                f"Using ASPP with {self.aspp_in_channels} in channels and {self.aspp_out_channels} out channels")
            self.aspp = ASPP(
                in_channels=self.aspp_in_channels,
                out_channels=self.aspp_out_channels,
                atrous_rates=self.atrous_rates,
            )

        # DUC Configuration
        duc_config = self.config["duc"]
        self.duc_enabled = duc_config.get("DUCEnabled", True)
        self.duc_up_factor = duc_config.get("DUCUpFactor", 2)

        # DUC module (only if enabled)
        if self.duc_enabled:
            print(f"Using DUC with upsampling factor {self.duc_up_factor}")
            self.duc = DUC(in_channels=self.aspp_out_channels,
                           up_factor=self.duc_up_factor)

        # 1x1 Convolution for segmentation mask output
        self.num_classes = self.config["model"]["NumClasses"]
        self.segmentation_head = nn.Conv2d(
            in_channels=self.aspp_out_channels, out_channels=self.num_classes, kernel_size=1
        )

    def forward(self, x):
        x = self.feature_extractor(x)  # ResNet-HDC backbone
        if self.aspp_enabled:
            x = self.aspp(x)  # ASPP for multi-scale context
        if self.duc_enabled:
            x = self.duc(x)  # Learned upsampling
        x = self.segmentation_head(x)  # 1x1 Convolution -> Segmentation mask
        return x
