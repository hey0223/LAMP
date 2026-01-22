import torch
import torch.nn.functional as F
import os
import copy
import logging
import numpy as np
from .actor_critic import Actor, Critic

class MADDPG:
    """
    Multi-Agent Deep Deterministic Policy Gradient (MADDPG) implementation.
    
    This class manages both government and household agents, handling action selection,
    network updates, and soft-target updates. It supports household embedding 
    concatenation and wealth-based sorting for observations to ensure consistent 
    critic inputs across different agent types.
    """
    def __init__(self, args, agent_id):
        """
        Initialize the MADDPG agent with actor and critic networks.

        Args:
            args (Namespace): Configuration containing hyperparameters and dimensions.
            agent_id (int): Unique identifier for the agent (last index indicates the government).
        """
        self.args = args
        self.agent_id = agent_id
        self.train_step = 0

        # Calculate input dimension for the Centralized Critic: 
        # includes government actions, household actions, and all agent observations.
        num_house = self.args.n_households
        house_priv_dim = self.args.house_obs_dim - self.args.embed_dim - self.args.gov_obs_dim
        assert house_priv_dim > 0, "house_priv_dim must be positive; check obs dim settings."
        critic_state_dim = num_house * (house_priv_dim + self.args.embed_dim) + self.args.gov_obs_dim
        critic_input = (
            self.args.gov_action_dim + num_house * self.args.house_action_dim + critic_state_dim
        )

        # Instantiate actor networks based on agent type (Government vs. Household)
        if agent_id == args.agent_block_num - 1:
            # Government agent: uses global observations and outputs government actions
            in_dim, out_dim = args.gov_obs_dim, args.gov_action_dim
        else:
            # Household agents: use private observations and output household actions
            in_dim, out_dim = args.house_obs_dim, args.house_action_dim
        
        self.actor_network = Actor(in_dim, out_dim, hidden_size=args.hidden_size)
        self.actor_target_network = Actor(in_dim, out_dim, hidden_size=args.hidden_size)

        # Instantiate the centralized critic network and its target
        self.critic_network = Critic(critic_input, hidden_size=args.hidden_size)
        self.critic_target_network = Critic(critic_input, hidden_size=args.hidden_size)

        # Synchronize target networks with the initial weights
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())

        # Initialize optimizers
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=args.p_lr)
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=args.q_lr)

        # Transfer networks to GPU if available
        if args.cuda:
            self.actor_network.cuda()
            self.critic_network.cuda()
            self.actor_target_network.cuda()
            self.critic_target_network.cuda()

    def select_action(self, obs_tensor, noise_rate, epsilon):
        """
        Select an action using an epsilon-greedy strategy with additive Gaussian noise.

        Args:
            obs_tensor (torch.Tensor): Observation input for the agent.
            noise_rate (float): Scaling factor for the Gaussian noise.
            epsilon (float): Probability of selecting a random exploration action.

        Returns:
            np.ndarray: The selected action, clipped to the range [-1, 1].
        """
        if np.random.rand() < epsilon:
            # Exploration: choose a random action from a uniform distribution
            dim = (self.args.gov_action_dim if self.agent_id == self.args.agent_block_num-1
                   else self.args.house_action_dim)
            action = np.random.uniform(-1, 1, dim)
        else:
            # Exploitation: use the actor network to predict the best action
            with torch.no_grad():
                pi = self.actor_network(obs_tensor).squeeze(0)
            action = pi.cpu().numpy()
            # Add Gaussian noise for exploration
            noise = noise_rate * np.random.randn(*action.shape)
            action = np.clip(action + noise, -1, 1)
        return action.copy()

    def _soft_update_target_network(self):
        """
        Update target network parameters using Polyak averaging (soft update).
        Target = (1 - tau) * Target + tau * Source
        """
        tau = self.args.tau
        # Update actor target network
        for targ, src in zip(self.actor_target_network.parameters(), self.actor_network.parameters()):
            targ.data.copy_((1 - tau) * targ.data + tau * src.data)
        # Update critic target network
        for targ, src in zip(self.critic_target_network.parameters(), self.critic_network.parameters()):
            targ.data.copy_((1 - tau) * targ.data + tau * src.data)

    def households_obs_sort(self, private_obs: torch.Tensor) -> torch.Tensor:
        """
        Generate sorting indices for households based on wealth (descending order).

        Args:
            private_obs (torch.Tensor): Tensor of household private observations.

        Returns:
            torch.Tensor: Indices that would sort households by wealth.
        """
        # Wealth is assumed to be at index 1 of the private observation vector
        return torch.argsort(private_obs[..., 1], dim=1, descending=True)

    def train(self, transitions, all_agents):
        """
        Perform a single training step for the actor and critic networks.

        This method handles:
        - Data conversion and device placement.
        - Wealth-based sorting for consistent agent representation.
        - Handling household groups and their respective rewards.
        - Computing target Q-values with centralized information.
        - Updating networks via backpropagation.

        Args:
            transitions (tuple): A batch of (glob_obs, priv_obs, gov_act, house_act, 
                                 gov_rew, house_rew, next_glob_obs, next_priv_obs, 
                                 done, curr_emb, next_emb).
            all_agents (list): List of all MADDPG agents in the environment.

        Returns:
            tuple: (actor_loss, critic_loss) as scalar values.
        """
        # Unpack transitions
        (glob_obs, priv_obs, gov_act, house_act,
         gov_rew, house_rew, next_glob_obs, next_priv_obs, done,
         curr_emb, next_emb) = transitions

        device = torch.device('cuda' if self.args.cuda else 'cpu')
        B, N = self.args.batch_size, self.args.n_households

        # Convert raw arrays to torch tensors
        g_obs = torch.tensor(glob_obs, dtype=torch.float32, device=device)
        p_obs = torch.tensor(priv_obs, dtype=torch.float32, device=device)
        g_act = torch.tensor(gov_act, dtype=torch.float32, device=device)
        h_act = torch.tensor(house_act, dtype=torch.float32, device=device)
        g_rew = torch.tensor(gov_rew, dtype=torch.float32, device=device).view(B, 1)
        h_rew = torch.tensor(house_rew, dtype=torch.float32, device=device).view(B, N, 1)
        ng_obs = torch.tensor(next_glob_obs, dtype=torch.float32, device=device)
        np_obs = torch.tensor(next_priv_obs, dtype=torch.float32, device=device)
        inv_done = (1 - torch.tensor(done, dtype=torch.float32, device=device)).unsqueeze(-1)
        
        # Use embeddings sampled from the replay buffer
        c_emb = torch.tensor(curr_emb, dtype=torch.float32, device=device)
        n_emb = torch.tensor(next_emb, dtype=torch.float32, device=device)

        # 1. Calculate wealth-based ranking indices for current time step
        idx = self.households_obs_sort(p_obs)          # Ranking at time t
        batch_idx = torch.arange(B).unsqueeze(1)
        
        # 2. Sort current time step data using calculated indices
        p_obs_sorted = p_obs[batch_idx, idx]
        h_act_sorted = h_act[batch_idx, idx]
        h_rew_sorted = h_rew[batch_idx, idx]
        emb_sorted = c_emb[batch_idx, idx]
        
        # 3. Sort next time step data using the same ranking indices to maintain agent consistency
        np_obs_sorted = np_obs[batch_idx, idx]
        n_emb_sorted = n_emb[batch_idx, idx]

        # 4. Construct complete observation vectors for all agents
        g_obs_rep = g_obs.unsqueeze(1).repeat(1, N, 1)
        ng_obs_rep = ng_obs.unsqueeze(1).repeat(1, N, 1)
        
        p_obs_full = torch.cat([g_obs_rep, p_obs_sorted, emb_sorted], dim=-1)
        np_obs_full = torch.cat([ng_obs_rep, np_obs_sorted, n_emb_sorted], dim=-1)

        # 5. Prepare Critic inputs, maintaining a consistent order: [Global, (P1, E1), (P2, E2), ...]
        pe_combined = torch.cat([p_obs_sorted, emb_sorted], dim=-1)
        next_pe_combined = torch.cat([np_obs_sorted, n_emb_sorted], dim=-1)
        
        critic_state = torch.cat([g_obs, pe_combined.view(B, -1)], dim=-1)
        next_critic_state = torch.cat([ng_obs, next_pe_combined.view(B, -1)], dim=-1)
        critic_action = torch.cat([h_act_sorted.view(B, -1), g_act], dim=-1)

        # Compute target Q-values
        with torch.no_grad():
            next_actions = []
            for aid in range(self.args.agent_block_num):
                net = all_agents[aid].actor_target_network
                if aid == self.args.agent_block_num - 1:
                    # Government next action
                    inp = ng_obs
                    next_act = net(inp)
                else:
                    # Household group next actions
                    if aid == 0: idxs = range(int(0.1 * N))
                    elif aid == 1: idxs = range(int(0.1 * N), int(0.5 * N))
                    else: idxs = range(int(0.5 * N), N)
                    inp = np_obs_full[:, idxs, :]
                    next_act = net(inp).view(B, -1)
                next_actions.append(next_act)
            next_u = torch.cat(next_actions, dim=1)

            # Target Q-value from the centralized target critic
            q_next = self.critic_target_network(next_critic_state, next_u) * inv_done
            
            # Determine reward for the current agent (Government vs. Household Group)
            if self.agent_id == self.args.agent_block_num - 1:
                reward = g_rew
            else:
                # Use mean reward for the corresponding household group
                if self.agent_id == 0: idxs = range(int(0.1 * N))
                elif self.agent_id == 1: idxs = range(int(0.1 * N), int(0.5 * N))
                else: idxs = range(int(0.5 * N), N)
                reward = h_rew_sorted[:, idxs, :].mean(dim=1)
                
            target_q = reward + self.args.gamma * q_next

        # Critic network update
        q_val = self.critic_network(critic_state, critic_action)
        critic_loss = F.mse_loss(q_val, target_q)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # Actor network update
        if self.agent_id == self.args.agent_block_num - 1:
            # Government policy update
            pred_act = self.actor_network(g_obs)
            actor_input_u = torch.cat([h_act_sorted.view(B, -1), pred_act], dim=1)
        else:
            # Household group policy update
            if self.agent_id == 0: idxs = range(int(0.1 * N))
            elif self.agent_id == 1: idxs = range(int(0.1 * N), int(0.5 * N))
            else: idxs = range(int(0.5 * N), N)
            
            group_obs = p_obs_full[:, idxs, :]
            pred_act = self.actor_network(group_obs)
            new_h_act = h_act_sorted.clone()
            new_h_act[:, idxs, :] = pred_act
            actor_input_u = torch.cat([new_h_act.view(B, -1), g_act], dim=1)
        
        # Policy gradient: maximize the critic's output for predicted actions
        actor_loss = -self.critic_network(critic_state, actor_input_u).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # Update target networks towards behavior networks
        self._soft_update_target_network()
        self.train_step += 1

        return actor_loss.item(), critic_loss.item()
