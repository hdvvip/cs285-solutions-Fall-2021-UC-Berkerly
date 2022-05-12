import abc
import itertools
from typing import Any
from torch import device, nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from cs285.infrastructure import pytorch_util as ptu
from cs285.policies.base_policy import BasePolicy


class MLPPolicy(BasePolicy, nn.Module, metaclass=abc.ABCMeta):

    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 discrete=False,
                 learning_rate=1e-4,
                 training=True,
                 nn_baseline=False,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # init vars
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.discrete = discrete
        self.size = size
        self.learning_rate = learning_rate
        self.training = training
        self.nn_baseline = nn_baseline

        if self.discrete:
            self.logits_na = ptu.build_mlp(
                input_size=self.ob_dim,
                output_size=self.ac_dim,
                n_layers=self.n_layers,
                size=self.size,
            )
            self.logits_na.to(ptu.device)
            self.mean_net = None
            self.logstd = None
            self.optimizer = optim.Adam(self.logits_na.parameters(),
                                        self.learning_rate)
        else:
            self.logits_na = None
            self.mean_net = ptu.build_mlp(
                input_size=self.ob_dim,
                output_size=self.ac_dim,
                n_layers=self.n_layers, size=self.size,
            )
            self.mean_net.to(ptu.device)
            self.logstd = nn.Parameter(
                torch.zeros(self.ac_dim, dtype=torch.float32, device=ptu.device)
            )
            self.logstd.to(ptu.device)
            self.optimizer = optim.Adam(
                itertools.chain([self.logstd], self.mean_net.parameters()),
                self.learning_rate
            )

    ##################################

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    ##################################

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None]
        # TODO return the action that the policy prescribes
        dist = self.forward(ptu.from_numpy(observation))
        return ptu.to_numpy(dist.rsample())

    # update/train this policy
    def update(self, observations, actions, **kwargs):
        raise NotImplementedError

    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, observation: torch.FloatTensor):
        if self.discrete:
            logits = self.logits_na(observation)
            action_distribution = distributions.Categorical(logits=logits)
            return action_distribution
        else:
            batch_mean = self.mean_net(observation)
            scale_tril = torch.diag(torch.exp(self.logstd))
            batch_dim = batch_mean.shape[0]
            batch_scale_tril = scale_tril.repeat(batch_dim, 1, 1)
            action_distribution = distributions.MultivariateNormal(
                batch_mean,
                scale_tril=batch_scale_tril,
            )
            return action_distribution


#####################################################
#####################################################

class MLPPolicySL(MLPPolicy):
    def __init__(self, ac_dim, ob_dim, n_layers, size, **kwargs):
        super().__init__(ac_dim, ob_dim, n_layers, size, **kwargs)
        self.loss = nn.MSELoss()

    def update(
            self, observations, actions,
            adv_n=None, acs_labels_na=None, qvals=None
    ):
        # TODO: update the policy and return the loss
        loss = None
        actions = ptu.from_numpy(actions)
        dist = self.forward(ptu.from_numpy(observations))
        curr_actions = dist.rsample()
        loss = self.loss(curr_actions, actions)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            # You can add extra logging information here, but keep this line
            'Training Loss': ptu.to_numpy(loss),
        }
  
# import abc
# import itertools
# from typing import Any
# from torch import nn
# from torch.nn import functional as F
# from torch import optim

# import numpy as np
# import torch
# from torch import distributions

# from cs285.infrastructure import pytorch_util as ptu
# from cs285.policies.base_policy import BasePolicy


# class MLPPolicy(BasePolicy, nn.Module, metaclass=abc.ABCMeta):

#     def __init__(self,
#                  ac_dim,
#                  ob_dim,
#                  n_layers,
#                  size,
#                  discrete=False,
#                  learning_rate=1e-4,
#                  training=True,
#                  nn_baseline=False,
#                  **kwargs
#                  ):
#         super().__init__(**kwargs)

#         # init vars
#         self.ac_dim = ac_dim
#         self.ob_dim = ob_dim
#         self.n_layers = n_layers
#         self.discrete = discrete
#         self.size = size
#         self.learning_rate = learning_rate
#         self.training = training
#         self.nn_baseline = nn_baseline

#         if self.discrete:
#             self.logits_na = ptu.build_mlp(
#                 input_size=self.ob_dim,
#                 output_size=self.ac_dim,
#                 n_layers=self.n_layers,
#                 size=self.size,
#             )
#             self.logits_na.to(ptu.device)
#             self.mean_net = None
#             self.logstd = None
#             self.optimizer = optim.Adam(self.logits_na.parameters(),
#                                         self.learning_rate)
#         else:
#             self.logits_na = None
#             self.mean_net = ptu.build_mlp(
#                 input_size=self.ob_dim,
#                 output_size=self.ac_dim,
#                 n_layers=self.n_layers, size=self.size,
#             )
#             self.mean_net.to(ptu.device)
#             self.logstd = nn.Parameter(
#                 torch.zeros(self.ac_dim, dtype=torch.float32, device=ptu.device)
#             )
#             self.logstd.to(ptu.device)
#             self.optimizer = optim.Adam(
#                 itertools.chain([self.logstd], self.mean_net.parameters()),
#                 self.learning_rate
#             )

#     ##################################

#     def save(self, filepath):
#         torch.save(self.state_dict(), filepath)

#     ##################################

#     def get_action(self, obs: np.ndarray) -> np.ndarray:
#         if len(obs.shape) > 1:
#             observation = obs
#         else:
#             observation = obs[None]

#         # TODO return the action that the policy prescribes
#         if self.discrete:
#             return ptu.to_numpy(self.logits_na(ptu.from_numpy(observation)))
#         else:
#             return ptu.to_numpy(self.mean_net(ptu.from_numpy(observation)))

#     # update/train this policy
#     def update(self, observations, actions, **kwargs):
#         raise NotImplementedError

#     # This function defines the forward pass of the network.
#     # You can return anything you want, but you should be able to differentiate
#     # through it. For example, you can return a torch.FloatTensor. You can also
#     # return more flexible objects, such as a
#     # `torch.distributions.Distribution` object. It's up to you!
#     def forward(self, observation: torch.FloatTensor) -> Any:
#         return self.logits_na(observation) if self.discrete else self.mean_net(observation)


# #####################################################
# #####################################################

# class MLPPolicySL(MLPPolicy):
#     def __init__(self, ac_dim, ob_dim, n_layers, size, **kwargs):
#         super().__init__(ac_dim, ob_dim, n_layers, size, **kwargs)
#         self.loss = nn.MSELoss()

#     def update(
#             self, observations, actions,
#             adv_n=None, acs_labels_na=None, qvals=None
#     ):
#         # TODO: update the policy and return the loss
#         # modified from https://pytorch.org/docs/stable/optim.html#taking-an-optimization-step
#         self.optimizer.zero_grad()
#         current_action = self.forward(ptu.from_numpy(observations))
#         loss = self.loss(current_action, ptu.from_numpy(actions))
#         loss.backward()
#         self.optimizer.step()
#         return {
#             # You can add extra logging information here, but keep this line
#             'Training Loss': ptu.to_numpy(loss),
#         }