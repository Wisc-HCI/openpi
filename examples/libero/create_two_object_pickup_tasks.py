import numpy as np

from libero.libero.utils.bddl_generation_utils import (
    get_xy_region_kwargs_list_from_regions_info,
)
from libero.libero.utils.mu_utils import InitialSceneTemplates, register_mu
from libero.libero.utils.task_generation_utils import (
    generate_bddl_from_task_info,
    register_task_info,
)


@register_mu(scene_type="tabletop")
class TwoObjectDiagonalPickup(InitialSceneTemplates):
    def __init__(self):
        # "table" 会在 BDDL 中生成：
        # main_table - table
        fixture_num_info = {
            "table": 1,
        }

        object_num_info = {
            "cream_cheese": 1,
            "tomato_sauce": 1,
        }

        super().__init__(
            workspace_name="main_table",
            fixture_num_info=fixture_num_info,
            object_num_info=object_num_info,
        )

    def define_regions(self):
        # 画面左上
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[-0.08, -0.08],
                region_name="cream_cheese_init_region",
                target_name=self.workspace_name,
                region_half_len=0.02,
            )
        )

        # 画面右下
        self.regions.update(
            self.get_region_dict(
                region_centroid_xy=[0.06, 0.06],
                region_name="tomato_sauce_init_region",
                target_name=self.workspace_name,
                region_half_len=0.02,
            )
        )

        self.xy_region_kwargs_list = (
            get_xy_region_kwargs_list_from_regions_info(self.regions)
        )

    @property
    def init_states(self):
        return [
            (
                "On",
                "cream_cheese_1",
                "main_table_cream_cheese_init_region",
            ),
            (
                "On",
                "tomato_sauce_1",
                "main_table_tomato_sauce_init_region",
            ),
        ]


def main():
    scene_name = "two_object_diagonal_pickup"
    objects = ["cream_cheese_1", "tomato_sauce_1"]

    register_task_info(
        language="pick up the cream cheese",
        scene_name=scene_name,
        objects_of_interest=objects,
        goal_states=[
            ("Up", "cream_cheese_1"),
        ],
    )

    register_task_info(
        language="pick up the tomato sauce",
        scene_name=scene_name,
        objects_of_interest=objects,
        goal_states=[
            ("Up", "tomato_sauce_1"),
        ],
    )

    output_dir = (
        "third_party/libero/libero/libero/"
        "bddl_files/custom_two_object_pickup"
    )

    files, failures = generate_bddl_from_task_info(folder=output_dir)

    if failures:
        raise RuntimeError(f"Failed to generate: {failures}")

    print("Generated:")
    for path in files:
        print(path)


if __name__ == "__main__":
    main()