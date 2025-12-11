import torch
from torch import nn

class FeatModel(nn.Module):
    def __init__(self, hp: dict):
        """
        initializer of the FeatModel
        :param hp: dictionary that includes all hyperparameters
        """
        super(FeatModel, self).__init__()
        self.hp = hp
        out_dims = self.hp['hidden_dims'][1::] + [self.hp['out_dim']]
        self.check_validity(out_dims)
        self.get_layers(out_dims)

    def get_layers(self, out_dims: list) -> None:
        """
        Method is used to define the layers of the model (dynamically according to the design choices)
        :param out_dims: list of the output dimensions per layer
        :return: None
        """
        for layer_idx in range(len(self.hp['hidden_dims'])):
            self.add_module(f'linear_{layer_idx}', nn.Linear(self.hp['hidden_dims'][layer_idx], out_dims[layer_idx]))
            if not layer_idx == len(self.hp['hidden_dims']) - 1:
                self.add_module(f'relu_{layer_idx}', nn.ReLU())
                self.add_module(f'dropout_{layer_idx}', nn.Dropout(p=self.hp['dropout'][layer_idx]))

    def check_validity(self, out_dims: list) -> None:
        """
        Method is used to validate the design choices for the model
        :param out_dims: list of the output dimensions per layer
        :return: None
        """
        assert len(out_dims) == len(self.hp['hidden_dims']), 'They must be in same size'
        assert len(out_dims) == len(self.hp['dropout'])


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Method is used for the forward pass of the model
        :param x: input data
        :return: output of the model
        """
        for each in self.children():
            out = each(x)
            x = out

        return x

