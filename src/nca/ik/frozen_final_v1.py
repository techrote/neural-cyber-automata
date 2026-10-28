"""Candidate-independent final-only recipe for frozen CPU inference v1."""
from dataclasses import dataclass, field
from typing import Literal
from nca.contracts import ContractError
from .data import DatasetConfig, GenerationBudget
from .reference import SolverConfig


@dataclass(frozen=True)
class FrozenFinalConfig(DatasetConfig):
    id: str = 'runner-frozen-cpu-final-v1'
    generation_seed: int = 53029
    family_namespace: str = 'runner-frozen-cpu-final-v1'
    ordinary_hip_lane: tuple[float,float] = (.54,.58)
    withheld_hip_lane: tuple[float,float] = (.96,.98)
    hip_lanes: tuple[tuple[float,float],...] = ((.06,.50),(.62,.70),(.975,.98))
    schema: Literal['nca-ik-frozen-final-config/v1'] = 'nca-ik-frozen-final-config/v1'
    budget: GenerationBudget = field(default_factory=lambda: GenerationBudget(
        max_unique_tasks=2940,max_teacher_attempts=2940,max_teacher_iterations=188160))

    @property
    def expected_tasks(self): return 2940

    def validate(self):
        super().validate()
        # No caller-controlled sampler search or count/teacher changes.
        if (self.id!='runner-frozen-cpu-final-v1' or self.generation_seed!=53029
                or self.family_namespace!='runner-frozen-cpu-final-v1'
                or self.ordinary_hip_lane!=(.54,.58) or self.withheld_hip_lane!=(.96,.98)
                or self.hip_lanes!=((.06,.50),(.62,.70),(.975,.98))
                or self.mode!='prospective' or self.points_per_bin!=(60,15,25)
                or self.trajectories_per_partition!=40 or self.trajectory_steps!=20
                or self.stress_per_class!=20 or self.batch_size!=64
                or self.withheld_bins!=(6,15,24,33,46,55,64,73)
                or self.joint_guard_radians!=.002 or self.point_guard_cells!=.0005
                or self.teacher!=SolverConfig(tolerance_cells=1e-7)
                or self.schema!='nca-ik-frozen-final-config/v1'
                or self.budget!=GenerationBudget(max_unique_tasks=2940,max_teacher_attempts=2940,max_teacher_iterations=188160)):
            raise ContractError('ik_frozen_recipe','fixed new final-only recipe required')

    def bounds(self,partition,bin_index):
        if partition!='final' or not 0<=bin_index<80:
            raise ContractError('ik_frozen_partition','new recipe is final-only')
        h,a=divmod(bin_index,8)
        lane=self.withheld_hip_lane if bin_index in self.withheld_bins else self.ordinary_hip_lane
        return ((-.6+.12*(h+lane[0]),-.6+.12*(h+lane[1])),
                (-.2+.05*(a+.08),-.2+.05*(a+.92)))


# Copyright 2026 techrote. All rights reserved.
