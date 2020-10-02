import torch
from torch import nn

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std, action_type):
        super(ActorCritic, self).__init__()

        # action mean range -1 to 1
        self.actor =  nn.Sequential(

                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 32),
                nn.Tanh(),
                nn.Linear(32, action_dim),
                nn.Softmax(dim=-1) if action_type == 'Discrete' else nn.Tanh()
        )

        self.fc_actor = nn.Sequential(
                nn.Linear(64*7*7, 512),
                nn.ReLU(),
                nn.Linear(512, action_dim),
                nn.Softmax(dim=-1)
        )

        # critic
        self.critic = nn.Sequential(

                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 32),
                nn.Tanh(),
                nn.Linear(32, 1)
        )

        self.fc_critic = nn.Sequential(
                nn.Linear(64*7*7, 512),
                nn.ReLU(),
                nn.Linear(512, 1)
        )

    def forward(self, x):
        raise NotImplementedError
        '''
        x, t = x
        if t == 'actor':
            x = self.actor(x)
            x = x.view(x.size(0), -1)
            x = self.fc_actor(x)
        elif t == 'critic':
            x = self.critic(x)
            x = x.view(x.size(0), -1)
            x = self.fc_critic(x)
        return x
        '''