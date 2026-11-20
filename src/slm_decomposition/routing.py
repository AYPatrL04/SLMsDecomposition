"""Build public-source prompts for the paired final study.

The controller orders facts using holder links. Only the checker and the model
perform arithmetic. A child-local S index always identifies a fact in that
child's input. R indices identify original document rows.
"""

import re
from collections import defaultdict

from . import prompts
from .source_check import parse_fact, query

ROLES = ("left", "right")
CHUNK_UPDATES = 4
INVALID_CITATION = 999
CITED_PREFIX = re.compile(r"(?im)^\s*\[?S([1-9]\d*)\]?\s*:")


def grouped_sources(case):
    """Find disjoint components for the two queried holders, in mention order."""
    parsed = query(case)
    if not parsed:
        return None
    unit, first, second, _ = parsed
    links = defaultdict(set)
    owners_by_fact = []
    for line in case["document"].splitlines():
        fact = parse_fact(line)
        if not fact or fact.unit != unit:
            return None
        owners = {fact.subject}
        if fact.other:
            owners.add(fact.other)
        for owner in owners:
            links[owner].update(owners)
        owners_by_fact.append(owners)

    groups = []
    for target in (first, second):
        if target not in links:
            return None
        pending = [target]
        reached = set()
        while pending:
            holder = pending.pop()
            if holder not in reached:
                reached.add(holder)
                pending.extend(links[holder] - reached)
        ids = [i for i, owners in enumerate(owners_by_fact, 1) if reached & owners]
        groups.append((reached, ids))
    left, right = (names for names, _ in groups)
    if left & right or left | right != set(links):
        return None
    return groups


def component_layout(case):
    """Order each unbranched path away from its unique anchor."""
    groups = grouped_sources(case)
    if not groups:
        return None
    lines = case["document"].splitlines()
    result = []
    for names, ids in groups:
        anchors = []
        edges = defaultdict(list)
        for i in ids:
            fact = parse_fact(lines[i - 1])
            if fact.other is None:
                anchors.append((i, fact.subject))
            else:
                edges[fact.subject].append((i, fact.other))
                edges[fact.other].append((i, fact.subject))
        if len(anchors) != 1:
            return None
        anchor, holder = anchors[0]
        if len(edges[holder]) > 1 or any(len(neighbors) > 2 for neighbors in edges.values()):
            return None
        path = [anchor]
        visited = {holder}
        while True:
            next_links = [(i, other) for i, other in edges[holder] if other not in visited]
            if not next_links:
                break
            i, holder = next_links[0]
            path.append(i)
            visited.add(holder)
        if set(path) != set(ids):
            return None
        result.append((names, path))
    return result


def indices(case, layout, slot=None):
    if layout is None:
        return list(range(1, len(case["document"].splitlines()) + 1))
    if slot:
        owner, unit = slot
        if owner and unit and unit.casefold() == query(case)[0]:
            for names, ids in layout:
                if owner.casefold() in names:
                    return ids
    return [i for _, ids in layout for i in ids]


def remap(text, source, destination):
    """Relabel step lines through the original document's citation IDs."""
    inverse = {raw_id: j for j, raw_id in enumerate(destination, 1)}

    def replace(match):
        index = int(match[1])
        target = inverse.get(source[index - 1]) if 1 <= index <= len(source) else None
        return f"S{target or INVALID_CITATION}:"

    return CITED_PREFIX.sub(replace, text)


def original_trace(case, text, stage, plan=None):
    """Restore original IDs for offline certification of a generated trace."""
    layout = component_layout(case)
    if not layout:
        return text
    slot = prompts.plan_slots(plan)[int(stage[-1]) - 1] if stage in ("child1", "child2") else None
    return remap(text, indices(case, layout, slot), indices(case, None))


def component(case, slot):
    """Locate a four-update boundary for a planned holder, without computing values."""
    owner, unit = slot
    layout = component_layout(case)
    if not layout or not owner or not unit or unit.casefold() != query(case)[0]:
        return None
    ids = next((ids for names, ids in layout if owner.casefold() in names), [])
    if len(ids) != 1 + 2 * CHUNK_UPDATES:
        return None
    lines = case["document"].splitlines()
    holder = parse_fact(lines[ids[0] - 1]).subject
    holders = [holder]
    for i in ids[1:]:
        fact = parse_fact(lines[i - 1])
        holder = fact.other if fact.subject == holder else fact.subject
        holders.append(holder)
    if holder != owner.casefold():
        return None
    boundary = holders[CHUNK_UPDATES]
    match = re.search(
        r"(?<![\w'’-])" + re.escape(boundary) + r"(?![\w'’-])", case["document"], re.IGNORECASE
    )
    return {"owner": owner, "unit": unit, "bridge": match[0] if match else boundary, "ids": ids}


def _insert(text, description):
    if not text or not description:
        return text
    return text.replace("Question: ", description + "\nQuestion: ", 1)


def _marker(case):
    parsed = query(case)
    if not parsed:
        return ""
    descriptions = []
    for side, owner in zip(ROLES, parsed[1:3]):
        part = component(case, (owner, parsed[0]))
        if not part:
            return ""
        ids = part["ids"]
        early = ", R".join(map(str, ids[: CHUNK_UPDATES + 1]))
        late = ", R".join(map(str, ids[CHUNK_UPDATES + 1 :]))
        descriptions.append(f"{side} component: R{early} to {part['bridge']}, then R{late}")
    return "Public chunk locations in original source IDs: " + " | ".join(descriptions) + "."


def _unordered_prompt(stage, case, previous, feedback):
    """Use raw-order grouping for the planner or when a path layout is unavailable."""
    text = prompts.make_prompt(stage, case, previous, feedback=feedback)
    groups = grouped_sources(case)
    if not text or not groups:
        return text
    if stage in ("child1", "child2"):
        owner, unit = prompts.plan_slots(previous["plan"]["text"])[int(stage[-1]) - 1]
        if owner and unit and unit.casefold() == query(case)[0]:
            selected = next((ids for names, ids in groups if owner.casefold() in names), [])
            if selected:
                lines = case["document"].splitlines()
                whole = "\n".join(f"S{i}: {line}" for i, line in enumerate(lines, 1))
                subset = "\n".join(f"S{i}: {lines[i - 1]}" for i in selected)
                return text.replace(whole, subset, 1)
    labels = "\n".join(
        f"Group {j}: " + ", ".join(f"S{i}" for i in ids) for j, (_, ids) in enumerate(groups, 1)
    )
    return _insert(text, "Connected source statements (shared people and items):\n" + labels)


def _ordered_prompt(stage, case, previous, feedback=True):
    layout = component_layout(case)
    if not layout or stage == "plan":
        return _unordered_prompt(stage, case, previous, feedback)
    slots = prompts.plan_slots(previous["plan"]["text"]) if "plan" in previous else ()
    is_child = stage in ("child1", "child2")
    ids = indices(case, layout, slots[int(stage[-1]) - 1] if is_child else None)
    lines = case["document"].splitlines()
    selected = {**case, "document": "\n".join(lines[i - 1] for i in ids)}
    mapping = "Original sentence IDs: " + " ".join(f"S{j}=R{i}" for j, i in enumerate(ids, 1)) + "."
    if not is_child:
        n = len(layout[0][1])
        mapping += f" Source components: S1–S{n} and S{n + 1}–S{len(ids)}."
    translated = previous.copy()
    if stage in ("full", "compact"):
        for j, slot in enumerate(slots, 1):
            key = f"child{j}"
            translated[key] = {
                "text": remap(previous[key]["text"], indices(case, layout, slot), ids)
            }
    return _insert(prompts.make_prompt(stage, selected, translated, feedback=feedback), mapping)


def _chunk_prompt(case, previous, side, phase):
    role = ROLES.index(side)
    slot = prompts.plan_slots(previous["plan"]["text"])[role]
    part = component(case, slot)
    if not part:
        return None if phase == "first" else _ordered_prompt(f"child{role + 1}", case, previous)
    ids = part["ids"]
    lines = case["document"].splitlines()
    if phase == "first":
        owner = part["bridge"]
        head = ids[: CHUNK_UPDATES + 1]
        subset = [lines[i - 1] for i in head]
        mapping = (
            "Original source IDs: " + " ".join(f"S{j}=R{i}" for j, i in enumerate(head, 1)) + "."
        )
    else:
        owner = part["owner"]
        first = previous[f"{side}_first"]["text"]
        carry = prompts.child_value(first)
        if (
            not carry
            or carry[0].casefold() != part["bridge"].casefold()
            or carry[2].casefold() != part["unit"].casefold()
        ):
            return None
        tail = ids[CHUNK_UPDATES + 1 :]
        subset = [f"{part['bridge']} has {carry[1]} {part['unit']}."]
        subset.extend(lines[i - 1] for i in tail)
        cited = sorted({int(k) for k in CITED_PREFIX.findall(first) if int(k) <= CHUNK_UPDATES + 1})
        provenance = ", ".join(f"R{ids[k - 1]}" for k in cited) or "no recognized source ID"
        mapping = (
            f"S1 carries the previous subproblem's generated amount for {part['bridge']}. "
            f"It cited {provenance}. Original IDs: "
            + " ".join(f"S{j}=R{i}" for j, i in enumerate(tail, 2))
            + "."
        )
    plan = f"Q1: {owner} | {part['unit']}\nQ2: {owner} | {part['unit']}"
    child = {**case, "document": "\n".join(subset)}
    return _insert(prompts.make_prompt("child1", child, {"plan": {"text": plan}}), mapping)


def combined(case, previous, side):
    """Join split generations in component-local IDs; a carry is not new evidence."""
    role = ROLES.index(side)
    slot = prompts.plan_slots(previous["plan"]["text"])[role]
    last = previous[f"{side}_last"]["text"]
    if not component(case, slot):
        return last
    prefix = re.compile(r"^\s*\[?S([0-9]+)\]?\s*:", re.IGNORECASE | re.MULTILINE)
    chunk_facts = CHUNK_UPDATES + 1

    def early_id(match):
        index = int(match[1])
        return f"S{index if 1 <= index <= chunk_facts else INVALID_CITATION}:"

    first = prefix.sub(early_id, previous[f"{side}_first"]["text"])
    continuation = []
    for line in last.splitlines():
        match = prefix.match(line)
        if match:
            index = int(match[1])
            if index == 1:
                continue
            label = index + CHUNK_UPDATES if 2 <= index <= chunk_facts else INVALID_CITATION
            line = f"S{label}:" + line[match.end() :]
        continuation.append(line)
    return first + "\nContinued working from the generated carry:\n" + "\n".join(continuation)


def make_prompt(stage, case, previous=None):
    """Build a registered stage prompt from public input and available generations."""
    previous = previous or {}
    if stage == "frozen_direct":
        stage = "direct"
    feedback = not stage.endswith("_no_check")
    stage = stage.removesuffix("_no_check")
    if stage in ("plan", "direct", "review"):
        text = _ordered_prompt(stage, case, previous, feedback)
    elif stage in ("flat_left", "flat_right"):
        role = ROLES.index(stage.removeprefix("flat_"))
        text = _ordered_prompt(f"child{role + 1}", case, previous)
    elif stage in ("left_first", "right_first", "left_last", "right_last"):
        return _chunk_prompt(case, previous, *stage.split("_"))
    elif stage in ("full", "flat_full", "compact"):
        children = {"plan": previous["plan"]}
        for j, side in enumerate(ROLES, 1):
            working = (
                previous[f"flat_{side}"]["text"]
                if stage == "flat_full"
                else combined(case, previous, side)
            )
            children[f"child{j}"] = {"text": working}
        text = _ordered_prompt("full" if stage == "flat_full" else stage, case, children, feedback)
    else:
        raise ValueError(f"Unknown study stage: {stage}")
    return _insert(text, _marker(case))
