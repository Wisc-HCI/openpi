import pathlib

from openpi.instruction_likelihood import bddl

BDDL_ROOT = pathlib.Path("third_party/libero/libero/libero/bddl_files")


def test_spatial_candidate_sets_are_grounded_by_the_two_bowls():
    tasks = bddl.load_task_specs(BDDL_ROOT, "libero_spatial")
    candidate_sets = bddl.build_candidate_sets(tasks)

    assert len(tasks) == 10
    assert all(len(candidate.executable_task_ids) == 2 for candidate in candidate_sets.values())
    assert all(candidate.task_id in candidate.executable_task_ids for candidate in candidate_sets.values())
    assert tasks[0].language.startswith("pick up the black bowl")
    assert tasks[0].bddl_language.startswith("Pick the akita black bowl")


def test_between_layout_accepts_only_its_two_present_bowl_relations():
    tasks = bddl.load_task_specs(BDDL_ROOT, "libero_spatial")
    candidate_sets = bddl.build_candidate_sets(tasks)
    task_id = "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate"

    assert set(candidate_sets[task_id].executable_task_ids) == {
        task_id,
        "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate",
    }


def test_cross_scene_mismatch_has_disjoint_referenced_types():
    layout = bddl.load_task_specs(BDDL_ROOT, "libero_spatial")[0]
    mismatch = bddl.choose_cross_scene_mismatch(layout, bddl.load_task_specs(BDDL_ROOT, "libero_object"))

    assert mismatch.suite == "libero_object"
    assert mismatch.problem != layout.problem
