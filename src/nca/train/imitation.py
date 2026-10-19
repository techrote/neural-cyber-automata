"""Convert admitted training demonstrations without opening heldout data."""

from __future__ import annotations

from nca.contracts import WireAction
from nca.controllers.baseline import SkillGoal
from nca.controllers.learned import action_vector, features
from nca.data import Dataset


def clone_sequences(dataset: Dataset):
    episodes = []
    for record in dataset.episodes:
        if record.split != "training":
            continue
        observations, actions = [], []
        previous = WireAction(effort=0.0)
        start_tick = int(record.initial_observation["tick"])
        for sample in record.imitation_samples():
            context = sample["context"]
            if context is None or context.skill != "directed_hop":
                raise ValueError("first-skill imitation requires an explicit directed-hop context")
            if sample["intervention"] != "none":
                raise ValueError("this frozen autonomous-teacher screen excludes correction demonstrations")
            if sample["reset"]:
                if observations:
                    episodes.append((observations, actions))
                observations, actions = [], []
                previous = WireAction(effort=0.0)
                start_tick = int(sample["observation"]["tick"])
            goal = SkillGoal(context.goal_id, context.skill, context.target, context.pace,
                             context.tolerance, context.timeout_seconds)
            observations.append(features(sample["observation"], goal, previous=previous, start_tick=start_tick))
            previous = WireAction.from_dict(dict(sample["action"]))
            actions.append(action_vector(previous))
        if observations:
            episodes.append((observations, actions))
    if not episodes:
        raise ValueError("no eligible training-only demonstration sequences")
    return episodes
