"""Parse LIBERO BDDL files and construct physically executable instruction sets.

The candidate construction deliberately uses initial-state predicates rather than
surface similarity between instructions.  For a spatial task, an instruction is
executable in another task's initial layout when an object of the referred type is
present at the referring placement and all objects/fixtures needed by its goal are
present in the current layout.
"""

from __future__ import annotations

from collections.abc import Iterable
import dataclasses
import pathlib
import re


@dataclasses.dataclass(frozen=True)
class Fact:
    predicate: str
    arguments: tuple[str, ...]

    @property
    def signature(self) -> tuple[str, ...]:
        """Placement signature, excluding the placed object itself."""
        return (self.predicate.lower(), *(arg.lower() for arg in self.arguments[1:]))


@dataclasses.dataclass(frozen=True)
class TaskSpec:
    path: pathlib.Path
    suite: str
    # Canonical benchmark prompt, derived exactly as LIBERO's benchmark loader does.
    language: str
    # Informational :language field from BDDL; Spatial uses a different wording
    # (for example, "Pick the akita black bowl ...") and is not the training prompt.
    bddl_language: str
    problem: str
    fixtures: dict[str, str]
    objects: dict[str, str]
    objects_of_interest: tuple[str, ...]
    initial_facts: tuple[Fact, ...]
    goal_facts: tuple[Fact, ...]

    @property
    def task_id(self) -> str:
        return self.path.stem

    @property
    def target_object(self) -> str:
        if not self.objects_of_interest:
            raise ValueError(f"{self.path} has no :obj_of_interest entry")
        return self.objects_of_interest[0]

    @property
    def target_type(self) -> str:
        return self.objects[self.target_object]

    def placement_of(self, object_name: str) -> Fact:
        facts = [fact for fact in self.initial_facts if fact.arguments and fact.arguments[0] == object_name]
        if len(facts) != 1:
            raise ValueError(
                f"Expected exactly one initial placement for {object_name} in {self.path}, found {len(facts)}"
            )
        return facts[0]

    def referenced_entities(self) -> set[str]:
        entities: set[str] = set()
        for fact in self.goal_facts:
            entities.update(fact.arguments)
        entities.add(self.target_object)
        return entities


@dataclasses.dataclass(frozen=True)
class CandidateSet:
    task_id: str
    true_instruction: str
    executable_task_ids: tuple[str, ...]
    executable_instructions: tuple[str, ...]
    rejected: dict[str, str]


def _strip_comments(text: str) -> str:
    return re.sub(r";[^\n]*", "", text)


def _section(text: str, name: str) -> str:
    match = re.search(rf"\(\s*:{re.escape(name)}\b", text, flags=re.IGNORECASE)
    if match is None:
        return ""
    depth = 0
    for index in range(match.start(), len(text)):
        if text[index] == "(":
            depth += 1
        elif text[index] == ")":
            depth -= 1
            if depth == 0:
                return text[match.start() : index + 1]
    raise ValueError(f"Unbalanced BDDL section :{name}")


def _typed_entities(section: str) -> dict[str, str]:
    body = re.sub(r"^\(\s*:\w+|\)$", "", section.strip(), flags=re.IGNORECASE).strip()
    tokens = re.findall(r"[^\s()]+", body)
    result: dict[str, str] = {}
    pending: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == "-":
            if not pending or index + 1 >= len(tokens):
                raise ValueError(f"Malformed typed declaration: {section}")
            entity_type = tokens[index + 1]
            result.update(dict.fromkeys(pending, entity_type))
            pending.clear()
            index += 2
        else:
            pending.append(token)
            index += 1
    if pending:
        raise ValueError(f"Untyped entities {pending} in: {section}")
    return result


def _flat_atoms(section: str, *, excluded: Iterable[str] = ()) -> tuple[Fact, ...]:
    excluded_lower = {value.lower() for value in excluded}
    facts: list[Fact] = []
    for match in re.finditer(r"\(([^()]+)\)", section):
        tokens = match.group(1).split()
        if len(tokens) >= 2 and tokens[0].lower() not in excluded_lower and not tokens[0].startswith(":"):
            facts.append(Fact(tokens[0], tuple(tokens[1:])))
    return tuple(facts)


def parse_bddl(path: pathlib.Path | str) -> TaskSpec:
    path = pathlib.Path(path)
    text = _strip_comments(path.read_text(encoding="utf-8"))
    language_section = _section(text, "language")
    bddl_language = re.sub(r"^\(\s*:language\s*|\)$", "", language_section.strip(), flags=re.IGNORECASE).strip()
    problem_match = re.search(r"\(\s*problem\s+([^\s()]+)", text, flags=re.IGNORECASE)
    if problem_match is None:
        raise ValueError(f"Missing problem declaration in {path}")
    interest_section = _section(text, "obj_of_interest")
    interest_tokens = re.findall(r"[^\s()]+", interest_section)[1:]
    return TaskSpec(
        path=path,
        suite=path.parent.name,
        language=_canonical_benchmark_language(path.stem),
        bddl_language=bddl_language,
        problem=problem_match.group(1),
        fixtures=_typed_entities(_section(text, "fixtures")),
        objects=_typed_entities(_section(text, "objects")),
        objects_of_interest=tuple(interest_tokens),
        initial_facts=_flat_atoms(_section(text, "init")),
        goal_facts=_flat_atoms(_section(text, "goal"), excluded=("and",)),
    )


def _canonical_benchmark_language(task_id: str) -> str:
    """Mirror ``libero.benchmark.grab_language_from_filename`` without importing MuJoCo."""
    if not task_id or not task_id[0].isupper():
        return task_id.replace("_", " ")
    scene_start = task_id.find("SCENE")
    if scene_start < 0:
        return task_id.replace("_", " ")
    # LIBERO-100 removes e.g. ``KITCHEN_SCENE3_`` or ``KITCHEN_SCENE10_``.
    language_start = scene_start + (8 if "SCENE10" in task_id else 7)
    return task_id[language_start:].replace("_", " ")


def load_task_specs(bddl_root: pathlib.Path | str, suite: str) -> list[TaskSpec]:
    suite_dir = pathlib.Path(bddl_root) / suite
    paths = sorted(suite_dir.glob("*.bddl"))
    if not paths:
        raise FileNotFoundError(f"No .bddl files found in {suite_dir}")
    return [parse_bddl(path) for path in paths]


def _entity_type(task: TaskSpec, entity: str) -> str | None:
    return task.objects.get(entity, task.fixtures.get(entity))


def executable_in_layout(candidate: TaskSpec, layout: TaskSpec) -> tuple[bool, str]:
    """Return whether ``candidate`` has a grounded interpretation in ``layout``.

    LIBERO-Spatial uses a placement predicate on the target object to ground the
    spatial referring phrase.  We require that the layout contain an object of the
    same type at exactly that predicate/relation, then validate all goal entities by
    type.  This catches the common but invalid construction that places all other
    Spatial instructions into every scene.
    """
    target_placement = candidate.placement_of(candidate.target_object).signature
    layout_target_objects = [name for name, kind in layout.objects.items() if kind == candidate.target_type]
    available_placements = {layout.placement_of(name).signature for name in layout_target_objects}
    if target_placement not in available_placements:
        return False, "no matching target object at the instruction's referring placement"

    layout_types = set(layout.objects.values()) | set(layout.fixtures.values())
    for entity in candidate.referenced_entities():
        kind = _entity_type(candidate, entity)
        if kind is None:
            return False, f"candidate entity {entity!r} has no declared type"
        if kind not in layout_types:
            return False, f"required entity type {kind!r} is absent from the layout"
    return True, "validated by initial placement and required entity types"


def build_candidate_sets(tasks: Iterable[TaskSpec]) -> dict[str, CandidateSet]:
    tasks = list(tasks)
    result: dict[str, CandidateSet] = {}
    for layout in tasks:
        accepted: list[TaskSpec] = []
        rejected: dict[str, str] = {}
        for candidate in tasks:
            valid, reason = executable_in_layout(candidate, layout)
            if valid:
                accepted.append(candidate)
            else:
                rejected[candidate.task_id] = reason
        if layout.task_id not in {task.task_id for task in accepted}:
            raise ValueError(f"True task {layout.task_id} was rejected from its own layout")
        result[layout.task_id] = CandidateSet(
            task_id=layout.task_id,
            true_instruction=layout.language,
            executable_task_ids=tuple(task.task_id for task in accepted),
            executable_instructions=tuple(task.language for task in accepted),
            rejected=rejected,
        )
    return result


def choose_cross_scene_mismatch(layout: TaskSpec, other_suite_tasks: Iterable[TaskSpec]) -> TaskSpec:
    """Choose a deterministic instruction whose scene and object semantics are disjoint."""
    layout_types = set(layout.objects.values()) | set(layout.fixtures.values())
    eligible: list[TaskSpec] = []
    for task in other_suite_tasks:
        referenced_types = {_entity_type(task, name) for name in task.referenced_entities()}
        referenced_types.discard(None)
        if task.problem != layout.problem and referenced_types.isdisjoint(layout_types):
            eligible.append(task)
    if not eligible:
        raise ValueError(f"Could not find a semantically and scenically disjoint mismatch for {layout.task_id}")
    return sorted(eligible, key=lambda task: (task.suite, task.task_id))[0]
