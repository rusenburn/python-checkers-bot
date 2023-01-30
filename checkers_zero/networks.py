import torch.nn as nn
import torch as T
from abc import ABC,abstractmethod
import os
from .helpers import get_device

class NNBase(nn.Module,ABC):
    def __init__(self) -> None:
        super().__init__()
        # super().__init__()
    
    @abstractmethod
    def forward(self,state:T.Tensor)->tuple[T.Tensor,T.Tensor]:
        raise NotImplementedError()
    
    def save_model(self,path:str)->None:
        try:
            T.save(self.state_dict(),path)
        except:
            print(f"Could not save nn to path:{path}")
    
    def load_model(self,path:str)->None:
        try:
            self.load_state_dict(T.load(path))
            print(f"The nn was loaded from {path}")
        except:
            print(f"Could not load nn from {path}")

    def clone(self)->'NNBase':
        raise NotImplementedError()


class SharedResNetwork(NNBase):
    def __init__(self,
                 shape: tuple,
                 n_actions: int,
                 filters=128,
                 fc_dims=512,
                 n_blocks=5):

        super().__init__()
        self.shape = shape
        self.n_actions = n_actions
        self.filters = filters
        self.fc_dims = fc_dims
        self.n_blocks = n_blocks

        self._blocks = nn.ModuleList(
            [ResBlock(filters) for _ in range(n_blocks)])

        self._shared = nn.Sequential(
            nn.Conv2d(shape[0], filters, 3, 1, 1),
            *self._blocks)

        self._pi_head = nn.Sequential(
            nn.Conv2d(filters, filters, 3, 1, 1),
            nn.Flatten(),
            nn.Linear(shape[1]*shape[2]*filters, fc_dims),
            nn.ReLU(),
            nn.Linear(fc_dims, n_actions))

        self._wdl_head = nn.Sequential(
            nn.Conv2d(filters, filters, 3, 1, 1),
            nn.Flatten(),
            nn.Linear(shape[1]*shape[2]*filters, fc_dims),
            nn.ReLU(),
            nn.Linear(fc_dims, 3))
            
        device = get_device()
        self._blocks.to(device)
        self._shared.to(device)
        self._pi_head.to(device)
        self._wdl_head.to(device)

    def forward(self, state: T.Tensor) -> tuple[T.Tensor, T.Tensor]:
        shared: T.Tensor = self._shared(state)
        pi: T.Tensor = self._pi_head(shared)
        wdl_nums: T.Tensor = self._wdl_head(shared)
        probs: T.Tensor = pi.softmax(dim=-1)
        wdl_probs = wdl_nums.softmax(dim=-1)
        return probs, wdl_probs

    def clone(self) -> 'NNBase':
        cloned = SharedResNetwork(shape=self.shape,
            n_actions=self.n_actions,
            filters=self.filters,
            fc_dims =self.fc_dims,
            n_blocks=self.n_blocks)
        cloned.load_state_dict(self.state_dict())
        return cloned

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self._block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.Conv2d(channels, channels, 3, 1, 1),
        )
        self._se = SqueezeAndExcite(channels, squeeze_rate=4)

    def forward(self, state: T.Tensor) -> T.Tensor:
        initial = state
        output: T.Tensor = self._block(state)
        output = self._se(output, initial)
        output += initial
        output = output.relu()
        return output

class SqueezeAndExcite(nn.Module):
    def __init__(self, channels, squeeze_rate):
        super().__init__()
        self.channels = channels
        self.prepare = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
        )
        self._fcs = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, int(channels//squeeze_rate)),
            nn.ReLU(),
            nn.Linear(int(channels//squeeze_rate), channels*2)
        )

    def forward(self, state: T.Tensor, input_: T.Tensor) -> T.Tensor:
        shape_ = input_.shape
        prepared: T.Tensor = self.prepare(state)
        prepared = self._fcs(prepared)
        splitted = prepared.split(self.channels, dim=1)
        w: T.Tensor = splitted[0]
        b: T.Tensor = splitted[1]
        z = w.sigmoid()
        z = z.unsqueeze(-1).unsqueeze(-1).expand((-1, -
                                                  1, shape_[-2], shape_[-1]))
        b = b.unsqueeze(-1).unsqueeze(-1).expand((-1, -
                                                  1, shape_[-2], shape_[-1]))
        output = (input_*z) + b
        return output