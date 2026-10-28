"""Development trainer v1: original loss/Adam, cheaper checked receipts."""
from nca.train.ik import IKTrainer, LossWeights, LENGTH_CELLS
from nca.ik.models_v3 import build_model
from nca.ik.kinematics import TorchFK, chain_from_profile
from nca.ik.profiles import runner_right_sole_profile


class PreparedRows:
    """Tensorize an already authenticated training subset once, without refitting.

    Whole-batch grouping and adjacent-pair eligibility remain the original
    sampler's responsibility. CPU fixtures compare this to the original builder.
    """
    def __init__(self, context, corpus, samples):
        from nca.ik.data import runtime_features
        if not corpus.accepted or not {id(s) for s in samples}.issubset({id(s) for s in corpus.training}):
            raise ValueError("exact authenticated training rows required")
        self.samples, self.context, self.config_sha256 = samples, context, corpus.config.sha256
        torch, device = context.torch, context.device
        profile_hash = runner_right_sole_profile().sha256
        features, refs, poses, endpoints = [], [], [], []
        for sample in samples:
            task, label = sample.task, sample.label
            supported = task.invalid_specimen is None and task.inputs.profile_sha256 == profile_hash
            features.append(runtime_features(task.inputs) if supported else (0.,)*14)
            admitted = supported and all((*task.inputs.present, *task.inputs.measured,
                                          *task.inputs.active, *task.inputs.task_mask))
            poses.append(bool(admitted and label.pose_loss))
            endpoints.append(bool(admitted and label.endpoint_loss))
            refs.append(label.teacher.configuration if poses[-1] else (0., 0.))
        self.features = torch.tensor(features, dtype=torch.float32, device=device)
        self.refs = torch.tensor(refs, dtype=torch.float32, device=device)
        self.poses = torch.tensor(poses, dtype=torch.bool, device=device)
        self.endpoints = torch.tensor(endpoints, dtype=torch.bool, device=device)
        address = {(s.task.family_id, s.task.trajectory_id, s.task.state.augmentation,
                    s.task.state.step): i for i, s in enumerate(samples) if s.task.state.kind == "trajectory"}
        self.next = {}
        import math
        for key, i in address.items():
            j = address.get((*key[:3], key[3]+1))
            if (j is not None and endpoints[i] and endpoints[j]
                    and math.dist(samples[i].task.inputs.target_root, samples[j].task.inputs.target_root) <= .10):
                self.next[i] = j

    def batch(self, indexes):
        from collections import Counter
        from nca.train.ik import IKTrainingBatch
        torch, device = self.context.torch, self.context.device
        positions = {idx: pos for pos, idx in enumerate(indexes)}
        pairs = [(positions[i], positions[self.next[i]]) for i in indexes
                 if i in self.next and self.next[i] in positions]
        selected = torch.tensor(indexes, dtype=torch.int64, device=device)
        return IKTrainingBatch(self.features[selected], self.refs[selected], self.poses[selected],
                               self.endpoints[selected],
                               torch.tensor(pairs, dtype=torch.int64, device=device).reshape(-1, 2),
                               tuple(self.samples[i].task.id for i in indexes), self.config_sha256,
                               dict(sorted(Counter(self.samples[i].label.teacher.status for i in indexes).items())))


class DevTrainer(IKTrainer):
    def __init__(self, context, normalization, dataset_config, config):
        self.context = context
        self.config_sha256 = dataset_config.sha256
        self.normalization_sha256 = normalization.sha256
        self.weights = LossWeights()
        self.model = build_model(context, normalization, head=config.head, hidden_width=config.hidden_width)
        self.fk = TorchFK(chain_from_profile(runner_right_sole_profile()), context)
        self.optimizer = context.torch.optim.Adam(
            self.model.parameters(), lr=config.lr(0, 1), foreach=False, fused=False,
            capturable=context.device.type == "cuda")
        self.attempted_steps = self.committed_steps = self.sample_presentations = 0
        self.failed = False

    def step(self, batch):
        if self.failed:
            raise ValueError("failed development trainer cannot continue")
        self.attempted_steps += 1
        self.sample_presentations += len(batch.sample_ids)
        torch = self.context.torch
        try:
            if not bool((batch.pose_mask | batch.endpoint_mask).any()):
                raise ValueError("empty supervision")
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
            total, terms, prediction, geometry = self.loss(batch)
            if not bool(torch.isfinite(total)):
                raise ValueError("nonfinite loss")
            total.backward()
            norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1., error_if_nonfinite=True)
            self.optimizer.step()
            tensors = list(self.model.parameters())
            for state in self.optimizer.state.values():
                tensors.extend(v for v in state.values() if isinstance(v, torch.Tensor))
            if not bool(torch.stack([torch.isfinite(t).all() for t in tensors]).all()):
                raise ValueError("nonfinite model/optimizer")
            if any(t.device != self.context.device for t in tensors):
                raise ValueError("optimizer/model device mismatch")
            self.committed_steps += 1
            numbers = torch.stack([total.detach(), norm.detach(), *[v.detach() for v in terms.values()]]).cpu().tolist()
            return {"loss": numbers[0], "gradient_norm_before_clip": numbers[1],
                    "terms": dict(zip(terms, numbers[2:])), "rows": len(batch.sample_ids),
                    "committed_steps": self.committed_steps,
                    "sample_presentations": self.sample_presentations}
        except BaseException:
            self.failed = True
            raise

# Copyright 2026 techrote. All rights reserved.
