"""
Module to create a ResNet with basic blocks (like ResNet-18 or ResNet-34).
Will allow to extend the basic blocks with more intermediate channels
and to add basic blocks add the end of the stages.
"""

import torch
from torch import nn

from gromo.containers.growing_block import RestrictedConv2dGrowingBlock
from gromo.containers.growing_container import GrowingContainer
from gromo.modules.growing_normalisation import GrowingBatchNorm2d


class ResNetBasicBlock(GrowingContainer):
    def __init__(
        self,
        in_features: int = 3,
        out_features: int = 1000,
        device: torch.device | str | None = None,
        activation: nn.Module = nn.ReLU(),
        input_block_kernel_size: int = 3,
        output_block_kernel_size: int = 3,
        reduction_factor: float = 0.25,
    ) -> None:
        super().__init__(
            in_features=in_features, out_features=out_features, device=device
        )
        self.activation = activation.to(device)
        inplanes = 64
        self.pre_net = nn.Sequential(
            nn.Conv2d(
                in_features,
                inplanes,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
                device=self.device,
            ),
            nn.BatchNorm2d(inplanes, device=self.device),
            self.activation,
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.stages = nn.ModuleList()
        nb_stages = 4
        for i in range(nb_stages):
            # for the future we could remove the basic block of the first stage
            # as there is no dowsampling
            stage = nn.Sequential()
            input_channels = inplanes * (2 ** max(0, i - 1))
            output_channels = inplanes * (2**i)
            hidden_channels = int(inplanes * (2**i) * reduction_factor)

            stage.append(
                RestrictedConv2dGrowingBlock(
                    in_channels=input_channels,
                    out_channels=output_channels,
                    hidden_channels=hidden_channels,
                    kwargs_first_layer={
                        "kernel_size": input_block_kernel_size,
                        "padding": 1,
                        "use_bias": False,
                        "stride": 2 if i > 0 else 1,
                    },
                    kwargs_layer={
                        "kernel_size": output_block_kernel_size,
                        "padding": 1,
                        "use_bias": False,
                    },
                    pre_activation=nn.Sequential(
                        nn.BatchNorm2d(input_channels, device=self.device),
                        self.activation,
                    ),
                    mid_activation=nn.Sequential(
                        GrowingBatchNorm2d(hidden_channels, device=self.device),
                        self.activation,
                    ),
                    extended_mid_activation=self.activation,
                    name=f"Stage {i} Block 0",
                    downsample=(
                        nn.Sequential(
                            nn.BatchNorm2d(input_channels, device=self.device),
                            self.activation,
                            nn.Conv2d(
                                in_channels=input_channels,
                                out_channels=output_channels,
                                kernel_size=1,
                                stride=2,
                                bias=False,
                                device=self.device,
                            ),
                        )
                        if i > 0
                        else torch.nn.Identity()
                    ),
                    device=self.device,
                )
            )
            self.stages.append(stage)

        self.post_net = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(
                inplanes * (2 ** (nb_stages - 1)), out_features, device=self.device
            ),
        )

    def append_block(
        self,
        stage_index: int = 0,
        input_block_kernel_size: int = 3,
        output_block_kernel_size: int = 3,
    ) -> None:
        """
        Append a new block to the specified stage of the ResNet.
        """
        if stage_index < 0 or stage_index >= len(self.stages):
            raise IndexError(
                f"Stage {stage_index} is out of range. There are {len(self.stages)} stages."
            )
        stage = self.stages[stage_index]
        input_channels = stage[-1].out_features
        output_channels = input_channels
        hidden_channels = 0
        stage.append(
            RestrictedConv2dGrowingBlock(
                in_channels=input_channels,
                out_channels=output_channels,
                hidden_channels=hidden_channels,
                kwargs_first_layer={
                    "kernel_size": input_block_kernel_size,
                    "padding": 1,
                    "use_bias": False,
                    "stride": 1,
                },
                kwargs_layer={
                    "kernel_size": output_block_kernel_size,
                    "padding": 1,
                    "use_bias": False,
                },
                pre_activation=nn.Sequential(
                    nn.BatchNorm2d(input_channels, device=self.device),
                    self.activation,
                ),
                mid_activation=nn.Sequential(
                    GrowingBatchNorm2d(hidden_channels, device=self.device),
                    self.activation,
                ),
                extended_mid_activation=self.activation,
                name=f"Stage {stage_index} Block {len(stage)}",
                device=self.device,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre_net(x)
        for stage in self.stages:
            x = stage(x)
        x = self.post_net(x)
        return x

    def extended_forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre_net(x)
        for stage in self.stages:
            for block in stage:
                if hasattr(block, "extended_forward"):
                    x = block.extended_forward(x)
                else:
                    x = block(x)
        x = self.post_net(x)
        return x


if __name__ == "__main__":
    # set_device("cpu")
    from torchinfo import summary

    from gromo.utils.utils import global_device, set_device

    print(f"Using device {global_device()}")

    # Example usage
    model = ResNetBasicBlock(out_features=10)
    print(model)

    # Create a dummy input tensor
    input_tensor = torch.randn(
        1, 3, 224, 224, device=global_device()
    )  # Batch size of 1, 3 channels, 224x224 image
    # output = model(input_tensor)
    # print(output.shape)  # Should be (1, 1000) for the output layer

    summary(model, input_size=(1, 3, 224, 224))

    # model.stages[0][0].init_computation()
    #
    # model.zero_grad()
    # output = model(input_tensor)
    # error = (output ** 2).sum()
    # error.backward()
    # model.stages[0][0].update_computation()
    #
    # model.stages[0][0].compute_optimal_updates()
    #
    # model.stages[0][0].scaling_factor = 1.0
    #
    # _ = model.extended_forward(input_tensor)
    #
    # model.stages[0][0].apply_change()

    model.append_block()
    print(model)

    empty_block = model.stages[0][1]
    empty_block.init_computation()

    print(empty_block.first_layer.__str__(verbose=2))
    print(empty_block.second_layer.__str__(verbose=2))

    model.zero_grad()
    output = model(input_tensor)
    error = (output**2).sum()
    error.backward()

    print(empty_block.first_layer.input.shape)
    print(empty_block.second_layer.pre_activity.shape)

    empty_block.update_computation()
    print(empty_block.second_layer.__str__(verbose=2))

    empty_block.compute_optimal_updates()
    empty_block.scaling_factor = 1.0

    _ = model.extended_forward(input_tensor)

    empty_block.apply_change()
