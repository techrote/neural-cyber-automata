"""Fixed, versioned final-only geometry recipe for the polar follow-up."""
from dataclasses import dataclass, field
from typing import Literal
from nca.contracts import ContractError
from nca.ik.data import DatasetConfig, GenerationBudget
from nca.ik.reference import SolverConfig


@dataclass(frozen=True)
class FreshFinalConfig(DatasetConfig):
    id: str = "runner-polar-fresh-final-v2"
    generation_seed: int = 53023
    hip_lanes: tuple[tuple[float,float], ...] = ((.06,.50),(.62,.70),(.74,.78))
    withheld_hip_lane: tuple[float,float] = (.02,.06)
    family_namespace: str = "runner-polar-fresh-v2"
    recipe: Literal["ordinary-074-078_withheld-002-006/v2"] = "ordinary-074-078_withheld-002-006/v2"
    schema: Literal["nca-ik-fresh-final-config/v2"] = "nca-ik-fresh-final-config/v2"
    budget: GenerationBudget = field(default_factory=lambda: GenerationBudget(
        max_unique_tasks=2940,max_teacher_attempts=2940,max_teacher_iterations=188160))

    @property
    def expected_tasks(self):
        return 2940

    def validate(self):
        super().validate()
        if (self.generation_seed != 53023 or self.hip_lanes != ((.06,.50),(.62,.70),(.74,.78))
                or self.withheld_bins != (6,15,24,33,46,55,64,73) or self.mode != "prospective"
                or self.id != "runner-polar-fresh-final-v2"
                or self.teacher != SolverConfig(tolerance_cells=1e-7)
                or self.budget != GenerationBudget(max_unique_tasks=2940,
                    max_teacher_attempts=2940,max_teacher_iterations=188160)
                or self.withheld_hip_lane != (.02,.06) or self.family_namespace != "runner-polar-fresh-v2"
                or self.points_per_bin != (60,15,25) or self.trajectories_per_partition != 40
                or self.trajectory_steps != 20 or self.stress_per_class != 20
                or self.joint_guard_radians != .002 or self.point_guard_cells != .0005):
            raise ContractError("ik_fresh_recipe", "fixed prospective final recipe required")

    def bounds(self, partition, bin_index):
        if partition != "final" or not 0 <= bin_index < 80:
            raise ContractError("ik_fresh_partition", "fresh recipe is final-only")
        h,a=divmod(bin_index,8)
        lane=self.withheld_hip_lane if bin_index in self.withheld_bins else self.hip_lanes[2]
        return ((-.6+.12*(h+lane[0]),-.6+.12*(h+lane[1])),
                (-.2+.05*(a+.08),-.2+.05*(a+.92)))

# Copyright 2026 techrote. All rights reserved.
