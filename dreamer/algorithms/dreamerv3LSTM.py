import torch
import torch.nn as nn
import numpy as np
import wandb
import random
from dreamer.modules.model import RSSM, RewardModel, ContinueModel

from dreamer.modules.encoder import Encoder
from dreamer.modules.decoder import Decoder
from dreamer.modules.actor import Actor
from dreamer.modules.critic import Critic
from dreamer.utils.utils import (
    compute_lambda_values,
    create_normal_dist,
    DynamicInfos,
)
from dreamer.utils.buffer import ReplayBuffer


class DreamerV3:
    def __init__(self,
        agent_id,
        observation_shape,
        action_size,
        writer,
        device,
        config,
        LSTM, 
    ):
        self.agent_id = agent_id
        self.device = device
        self.action_size = action_size
        self.encoder = Encoder(observation_shape, config).to(self.device)
        self.target_encoder = Encoder(observation_shape, config).to(self.device)
        self.hard_update(self.target_encoder, self.encoder)
        self.decoder = Decoder(observation_shape, config).to(self.device)
        self.rssm = RSSM(action_size, config, LSTM).to(self.device)
        self.reward_predictor = RewardModel(config).to(self.device)
        self.hard_update(self.rssm.transition_model_target, self.rssm.transition_model)
        if config.parameters.dreamer.use_continue_flag:
            self.continue_predictor = ContinueModel(config).to(self.device)
        self.actor = Actor(action_size, config).to(self.device)
        self.critic = Critic(config).to(self.device)
        self.buffer = ReplayBuffer(observation_shape, action_size, self.device, config)

        self.config = config.parameters.dreamer

        # optimizer
        self.model_params = (
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.rssm.parameters())
            + list(self.reward_predictor.parameters())
        )
        if self.config.use_continue_flag:
            self.model_params += list(self.continue_predictor.parameters())

        self.model_optimizer = torch.optim.Adam(
            self.model_params, lr=self.config.model_learning_rate,
            weight_decay=0.0001
        )
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.config.actor_learning_rate,
            weight_decay=0.0001
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.config.critic_learning_rate,
            weight_decay=0.0001
        )
        self.continue_criterion = nn.BCELoss()

        self.dynamic_learning_infos = DynamicInfos(self.device)
        self.behavior_learning_infos = DynamicInfos(self.device)

        self.writer = writer
        self.num_total_episode = 0
    def hard_update(self, target, original):
        target.load_state_dict(original.state_dict())

    def soft_update(self, target, source, tau=0.005):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def train(self, metrics):
        for _ in range(self.config.train_iterations):
            for _ in range(self.config.collect_interval):
                data = self.buffer.sample(
                    self.config.batch_size, self.config.batch_length
                )
                posteriors, det = self.dynamic_learning(data, metrics)
                self.latent_imagination(posteriors, metrics)
        wandb.log(metrics)

    def dynamic_learning(self, data, metrics):
        prior, deterministic = self.rssm.recurrent_model_input_init(len(data.action))

        data.embedded_observation = self.encoder(data.observation, seq=1)
        with torch.no_grad():
            data.sg_embedding = self.target_encoder(data.observation, seq=1)
        for t in range(1, self.config.batch_length):
            deterministic = self.rssm.recurrent_model(
                prior, data.action[:, t - 1]
            )

            prior_dist, prior = self.rssm.transition_model(deterministic)
            with torch.no_grad():
                prior_dist_sg, prior_sg = self.rssm.transition_model_target(deterministic.detach())
                posterior_dist_sg, posterior_sg = self.rssm.representation_model(
                data.sg_embedding[:,t], deterministic
                )

            posterior_dist, posterior = self.rssm.representation_model(
                data.embedded_observation[:,t], deterministic
            )

            

            self.dynamic_learning_infos.append(
                priors=prior,
                posteriors=posterior,
                prior_dists_sg_mean = prior_dist_sg.mean,
                prior_dists_sg_std = prior_dist_sg.scale,

                posterior_dists_sg_std = posterior_dist_sg.scale,
                posterior_dists_sg_mean = posterior_dist_sg.mean,

                prior_dist_means=prior_dist.mean,
                prior_dist_stds=prior_dist.scale,
                posterior=posterior,
                posterior_dist_means=posterior_dist.mean,
                posterior_dist_stds=posterior_dist.scale,
                deterministics=deterministic,
            )

            prior = posterior

        infos = self.dynamic_learning_infos.get_stacked()
        self._model_update(data, infos, metrics)
        return infos.posteriors.detach(), infos.deterministics.detach()

    def _model_update(self, data, posterior_info, metrics):
        reconstructed_observation_dist = self.decoder(
            posterior_info.posteriors, posterior_info.deterministics, seq=1
        ) 
        reconstruction_observation_loss = reconstructed_observation_dist.log_prob(
            data.observation[:, 1:]
        )
        if self.config.use_continue_flag:
            continue_dist = self.continue_predictor(
                posterior_info.posteriors, posterior_info.deterministics
            )
            continue_loss = self.continue_criterion(
                continue_dist.probs, 1 - data.done[:, 1:]
            )

        reward_dist = self.reward_predictor(
            posterior_info.posteriors, posterior_info.deterministics
        )
        reward_loss = reward_dist.log_prob(data.reward[:, 1:])

        prior_dist = create_normal_dist(
            posterior_info.prior_dist_means,
            posterior_info.prior_dist_stds,
            event_shape=1,
        )
        posterior_dist = create_normal_dist(
            posterior_info.posterior_dist_means,
            posterior_info.posterior_dist_stds,
            event_shape=1,
        )
        prior_dist_sg = create_normal_dist(
            posterior_info.prior_dists_sg_mean,
            posterior_info.prior_dists_sg_std,
            event_shape=1,
        )
        posterior_dist_sg = create_normal_dist(
            posterior_info.posterior_dists_sg_mean,
            posterior_info.posterior_dists_sg_std,
            event_shape=1,
        )
        kl1 = torch.distributions.kl.kl_divergence(posterior_dist, prior_dist_sg)
        kl2 = torch.distributions.kl.kl_divergence(posterior_dist_sg, prior_dist)
        kl_divergence_loss = torch.max(torch.ones_like(kl1, device=kl1.device), kl1).mean()+torch.max(torch.ones_like(kl2, device=kl2.device), kl2).mean()
        model_loss = (
            self.config.kl_divergence_scale * kl_divergence_loss
            - reconstruction_observation_loss.mean()
            - reward_loss.mean()
        )
        if self.config.use_continue_flag:
            model_loss += continue_loss.mean()

        self.model_optimizer.zero_grad()
        model_loss.backward()
        nn.utils.clip_grad_norm_(
            self.model_params,
            self.config.clip_grad*10,
            norm_type=self.config.grad_norm_type,
        )
        self.model_optimizer.step()
        id = self.agent_id + 1
        metrics['dynamics_loss_' + str(id)] = kl1.mean().item()
        metrics['representation_loss_' + str(id)] = kl2.mean().item()
        metrics['representation_loss_'+str(id)] = -reconstruction_observation_loss.mean().item()
        metrics['reward_loss_'+str(id)] = -reward_loss.mean().item()
        metrics['continue_loss_'+str(id)] = -continue_loss.mean().item()

        self.soft_update(self.rssm.transition_model_target, self.rssm.transition_model)
        self.soft_update(self.target_encoder, self.encoder)

    def latent_imagination(self, states, metrics):
        state = states.reshape(-1, self.config.stochastic_size)
        size = self.config.batch_size*2
        indx = random.choices(list(range(state.shape[0])),k=size)
        state = state[indx]
        deterministic = self.rssm.recurrent_model.input_init(size)

        # continue_predictor reinit
        for t in range(self.config.horizon_length):
            action, log_prob = self.actor(state, deterministic)
            deterministic = self.rssm.recurrent_model(state, action)
            _, state = self.rssm.transition_model(deterministic)
            self.behavior_learning_infos.append(
                priors=state, deterministics=deterministic,
                actions=action, 
                log_probs = log_prob
            )

        self._agent_update(self.behavior_learning_infos.get_stacked(), metrics)
    def save_state_dict(self):
        self.rssm.recurrent_model.input_init(1)
        id = self.agent_id + 1
        torch.save(self.rssm.state_dict(), 'RSSM'+str(id))
        torch.save(self.encoder.state_dict(), 'ENCODER'+str(id))        
        torch.save(self.actor.state_dict(), 'ACTOR'+str(id))        
        torch.save(self.critic.state_dict(), 'CRITIC'+str(id))        
        torch.save(self.reward_predictor.state_dict(), 'REWARD'+str(id))        
        torch.save(self.continue_predictor.state_dict(), 'CONTINUE'+str(id))   

    def load_state_dict(self):
        id = self.agent_id + 1
        self.rssm.load_state_dict(torch.load('RSSM'+ str(id)) )
        self.encoder.load_state_dict(torch.load('ENCODER'+ str(id)))
        self.actor.load_state_dict(torch.load('ACTOR'+ str(id)))
        self.critic.load_state_dict(torch.load('CRITIC'+ str(id)))
        self.reward_predictor.load_state_dict(torch.load('REWARD'+ str(id)))
        self.continue_predictor.load_state_dict(torch.load('CONTINUE'+ str(id)))
    def _agent_update(self, behavior_learning_infos, metrics):
        predicted_rewards = self.reward_predictor(
            behavior_learning_infos.priors, behavior_learning_infos.deterministics
        ).mean
        values = self.critic(
            behavior_learning_infos.priors, behavior_learning_infos.deterministics
        ).mean

        if self.config.use_continue_flag:
            continues = self.continue_predictor(
                behavior_learning_infos.priors, behavior_learning_infos.deterministics
            ).mean
        else:
            continues = self.config.discount * torch.ones_like(values)

        lambda_values = compute_lambda_values(
            predicted_rewards,
            values,
            continues,
            self.config.horizon_length,
            self.device,
            behavior_learning_infos.log_probs,
            self.config.lambda_,
        )
        id = self.agent_id + 1
        actor_loss = -torch.mean(lambda_values)#-torch.log(1-actions**2+0.001).view(-1).mean()
        metrics["actor_loss_" + str(id)] = actor_loss.mean().item()

        self.actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(
            self.actor.parameters(),
            self.config.clip_grad,
            norm_type=self.config.grad_norm_type,
        )
        self.actor_optimizer.step()

        value_dist = self.critic(
            behavior_learning_infos.priors.detach()[:, :-1],
            behavior_learning_infos.deterministics.detach()[:, :-1],
        )
        value_loss = -torch.mean(value_dist.log_prob(lambda_values.detach()))#+torch.max(torch.ones_like(value_dist.mean), torch.abs((lambda_values.detach())-value_dist.mean)/value_dist.stddev.pow(2)).mean()
        metrics["critic_loss_" + str(id)] = value_loss.mean().item()

        self.critic_optimizer.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(
            self.critic.parameters(),
            self.config.clip_grad,
            norm_type=self.config.grad_norm_type,
        )
        self.critic_optimizer.step()

