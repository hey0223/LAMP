import torch
import numpy as np
from .actor_critic import Actor, Critic


class MADDPG:
    def __init__(self, args, agent_id):
        self.args = args
        self.agent_id = agent_id
        self.train_step = 0

        # Keep the pasted-version logic:
        # critic sees all sorted household private observations flattened, plus global obs.
        # Here, "private observations" are augmented with household embeddings.
        critic_input_size = (
            self.args.gov_action_dim
            + self.args.house_action_dim * self.args.n_households
            + self.args.gov_obs_dim
            + (self.args.house_obs_dim - self.args.gov_obs_dim) * self.args.n_households
        )

        # create the network
        if agent_id == self.args.agent_block_num - 1:   # government agent
            self.actor_network = Actor(args.gov_obs_dim, args.gov_action_dim, hidden_size=args.hidden_size)
            self.actor_target_network = Actor(args.gov_obs_dim, args.gov_action_dim, hidden_size=args.hidden_size)
        else:  # household agent
            # household actor input = [global_obs, private_obs, embed]
            self.actor_network = Actor(args.house_obs_dim, args.house_action_dim, hidden_size=args.hidden_size)
            self.actor_target_network = Actor(args.house_obs_dim, args.house_action_dim, hidden_size=args.hidden_size)

        self.critic_network = Critic(critic_input_size, hidden_size=args.hidden_size)
        self.critic_target_network = Critic(critic_input_size, hidden_size=args.hidden_size)

        # load the weights into the target networks
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())

        # create the optimizer
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.p_lr)
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args.q_lr)

        # if use the cuda...
        if self.args.cuda:
            self.actor_network.cuda()
            self.critic_network.cuda()
            self.actor_target_network.cuda()
            self.critic_target_network.cuda()

    def select_action(self, o, noise_rate, epsilon):
        if np.random.uniform() < epsilon:
            if self.agent_id == self.args.agent_block_num - 1:
                action_dim = self.args.gov_action_dim
            else:
                action_dim = self.args.house_action_dim
            u = np.random.uniform(-1, 1, action_dim)
        else:
            pi = self.actor_network(o).squeeze(0)
            u = pi.detach().cpu().numpy()
            # Keep the pasted-version exploration style.
            noise = noise_rate * np.random.rand(*u.shape)
            u += noise
            u = np.clip(u, -1, +1)
        return u.copy()

    # soft update
    def _soft_update_target_network(self):
        for target_param, param in zip(self.actor_target_network.parameters(), self.actor_network.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)

        for target_param, param in zip(self.critic_target_network.parameters(), self.critic_network.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)

    def households_obs_sort(self, private_obs):
        # Keep the pasted-version sorting rule: wealth is the last private-observation dimension.
        sorted_indices = torch.argsort(private_obs[:, :, -1], descending=True)
        return sorted_indices

    # update the network
    def train(self, transitions, all_agents):
        (
            global_obs,
            private_obs,
            gov_action,
            hou_action,
            gov_reward,
            house_reward,
            next_global_obs,
            next_private_obs,
            done,
            curr_emb,
            next_emb,
        ) = transitions

        device = torch.device('cuda' if self.args.cuda else 'cpu')

        global_obses = torch.tensor(global_obs, dtype=torch.float32, device=device)
        private_obses = torch.tensor(private_obs, dtype=torch.float32, device=device)
        gov_actions = torch.tensor(gov_action, dtype=torch.float32, device=device)
        hou_actions = torch.tensor(hou_action, dtype=torch.float32, device=device)
        gov_rewards = torch.tensor(gov_reward, dtype=torch.float32, device=device)
        house_rewards = torch.tensor(house_reward, dtype=torch.float32, device=device)
        next_global_obses = torch.tensor(next_global_obs, dtype=torch.float32, device=device)
        next_private_obses = torch.tensor(next_private_obs, dtype=torch.float32, device=device)
        inverse_dones = torch.tensor(1 - done, dtype=torch.float32, device=device).unsqueeze(-1)
        curr_embs = torch.tensor(curr_emb, dtype=torch.float32, device=device)
        next_embs = torch.tensor(next_emb, dtype=torch.float32, device=device)

        if gov_rewards.dim() == 1:
            gov_rewards = gov_rewards.unsqueeze(-1)
        if house_rewards.dim() == 2:
            house_rewards = house_rewards.unsqueeze(-1)

        num_set = range(0, self.args.n_households)
        num = []
        num.append(num_set[:int(0.1 * self.args.n_households)])
        num.append(num_set[int(0.1 * self.args.n_households):int(0.5 * self.args.n_households)])
        num.append(num_set[int(0.5 * self.args.n_households):])

        # Keep the pasted-version alignment rule:
        # one sorting index from current private obs, reused for current and next state.
        sorted_index = self.households_obs_sort(private_obses)
        batch_index = torch.arange(self.args.batch_size, device=device)[:, None]

        private_obses = private_obses[batch_index, sorted_index]
        hou_actions = hou_actions[batch_index, sorted_index]
        house_rewards = house_rewards[batch_index, sorted_index]
        next_private_obses = next_private_obses[batch_index, sorted_index]
        curr_embs = curr_embs[batch_index, sorted_index]
        next_embs = next_embs[batch_index, sorted_index]

        if self.agent_id == self.args.agent_block_num - 1:  # government agent
            r = gov_rewards.view(-1, 1)
        else:
            # Keep the pasted-version reward semantics:
            # household blocks keep per-household rewards instead of group mean.
            r = house_rewards[:, num[self.agent_id]]

        # Treat embed as extra private observation.
        private_aug = torch.cat((private_obses, curr_embs), dim=-1)
        next_private_aug = torch.cat((next_private_obses, next_embs), dim=-1)

        # Critic state follows the pasted version:
        # flatten all sorted household private features, then append global obs.
        o = torch.cat((private_aug.reshape(self.args.batch_size, -1), global_obses), dim=-1)

        # Household actor input follows the pasted version, with embed appended to private obs:
        # [global_obs, private_obs, embed]
        n_next_global_obses = next_global_obses.unsqueeze(1).repeat(1, self.args.n_households, 1)
        o_next = torch.cat((n_next_global_obses, next_private_aug), dim=-1)

        # calculate the target Q value function
        u_next = []
        with torch.no_grad():
            for agent_id in range(self.args.agent_block_num):
                if agent_id == self.args.agent_block_num - 1:
                    this_next_o = next_global_obses
                else:
                    this_next_o = o_next[:, num[agent_id]]
                u_next.append(all_agents[agent_id].actor_target_network(this_next_o).reshape(self.args.batch_size, -1))

            u_next = torch.cat(u_next, dim=1)
            flatten_o_next = torch.cat((next_private_aug.reshape(self.args.batch_size, -1), next_global_obses), dim=-1)
            q_next = self.critic_target_network(flatten_o_next, u_next).detach() * inverse_dones
            if self.agent_id == self.args.agent_block_num - 1:
                target_q = (r + self.args.gamma * q_next).detach()
            else:
                target_q = (
                    r + self.args.gamma * q_next.unsqueeze(2).repeat(1, len(num[self.agent_id]), 1)
                ).detach()

        # the q loss
        if self.agent_id == self.args.agent_block_num - 1:
            q_value = self.critic_network(o, torch.cat((hou_actions.reshape(self.args.batch_size, -1), gov_actions), dim=-1))
        else:
            q_value = self.critic_network(o, torch.cat((hou_actions.reshape(self.args.batch_size, -1), gov_actions), dim=-1)).unsqueeze(2).repeat(1, len(num[self.agent_id]), 1)

        critic_loss = (target_q - q_value).pow(2).mean()
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # the actor loss
        new_house_actions = hou_actions.clone()
        new_gov_actions = gov_actions.clone()
        if self.agent_id == self.args.agent_block_num - 1:
            new_gov_actions = self.actor_network(global_obses)
        else:
            this_o = torch.cat(
                (
                    global_obses.unsqueeze(1).repeat(1, len(num[self.agent_id]), 1),
                    private_aug[:, num[self.agent_id]],
                ),
                dim=-1,
            )
            new_house_actions[:, num[self.agent_id]] = self.actor_network(this_o)

        u = torch.cat((new_house_actions.view(self.args.batch_size, -1), new_gov_actions), dim=1)
        actor_loss = -self.critic_network(o, u).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        self._soft_update_target_network()
        self.train_step += 1

        return actor_loss.item(), critic_loss.item()
