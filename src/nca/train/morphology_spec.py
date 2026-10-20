"""Prospective morphology learner constants; importing them starts no model."""

from nca.data.morphology import FIXED_LATENT_STD, SAMPLER_SCHEMA

LEARNING_SPEC = {
    "schema": "nca-morphology-learning/v1", "sampling_distribution": SAMPLER_SCHEMA,
    "latent_standard_deviation": FIXED_LATENT_STD, "trainable_sampling_parameters": 0,
    "likelihood_space": "original active Gaussian latent coordinates and active Bernoulli semantic choices",
    "continuous_transform": "tanh for drive_x/drive_y/trim; sigmoid for effort, before common wire normalization",
    "inactive_channels": "fixed zero action/history; no imitation, actor or entropy contribution",
    "bc_epochs": 40, "bc_learning_rate": .001, "bc_discrete_weight": .25,
    "bc_positive_weight": "available negative/positive labels, positive denominator at least one, clamped [1,10]",
    "ppo_learning_rate": .0003, "ppo_epochs": 4, "ppo_clip_ratio": .2,
    "ppo_value_weight": .5, "ppo_entropy_weight": .001,
    "ppo_gamma_per_base_tick": .997, "ppo_lambda_per_base_tick": .99,
    "entropy_space": "active latent Gaussian plus active Bernoulli; not normalized wire entropy",
    "unroll": 16, "burn_in": "complete true episode prefix without gradient under current weights",
    "gradient_norm_maximum": .5, "padding": "no recurrence advance or loss on missing timesteps",
}
