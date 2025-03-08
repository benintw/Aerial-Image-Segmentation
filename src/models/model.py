import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

from typing import Any
import yaml

from icecream import ic


class RoadSegModel(nn.Module):

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.backbone = smp.Unet(
            encoder_name=config["encoder"],
            encoder_weights=config["weights"],
            in_channels=config["in_channels"],
            classes=config["classes"],
            activation=config["activation"],
        )  # outputs logits

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone(images)


if __name__ == "__main__":

    with open("configs/model.yaml", "r") as f:
        config = yaml.safe_load(f)

    model = RoadSegModel(config)
    # print(model)
    print(len(list(model.modules())))
    ic(list(model.modules()))
    # ic(list(model.backbone.modules()))

    random_input = torch.rand(size=(1, 3, 512, 512))
    output = model(random_input)
    print(output.shape)
