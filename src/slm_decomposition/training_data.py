"""Build joint source-grounded tasks exclusively from annotated training parents."""

import re

from .prompts import make_prompt, plan_slots


def render_fact(step, unit):
    sid, owner, value = step["sid"], step["owner"], int(step["value"])
    before = step["from"]
    if before is None:
        return f"S{sid}: {owner} = {value} {unit}"
    previous, delta = int(step["previous"]), int(step["delta"])
    if previous + delta != value:
        raise ValueError("Inconsistent training proof")
    op = "+" if delta >= 0 else "-"
    return f"S{sid}: {owner} = {before} {previous} {op} {abs(delta)} = {value} {unit}"


def child_text(child):
    lines = [render_fact(s, child["unit"]) for s in child["steps"]]
    return "\n".join(lines + [f"Value: {child['name']} = {child['value']} {child['unit']}"])


def tamper(child, variant, source_size):
    """Produce a noisy child transcript typical of operand mistakes or early completion."""
    c = dict(child)
    steps = [dict(s) for s in c["steps"]]
    if variant == 0:
        c["value"] += 1
    elif variant == 1:
        end = steps[-1]
        if end["delta"] is None:
            end["value"] += 2
        else:
            end["delta"] *= -1
            end["value"] = end["previous"] + end["delta"]
        c["value"] = end["value"]
    elif variant == 2:
        # The target amount is omitted while cited intermediate workings survive.
        return "\n".join(child_text(c).splitlines()[:-1])
    else:
        # The amount and source citation belong to different statements.
        steps[-1]["sid"] = steps[-1]["sid"] % source_size + 1
        c["value"] += 3
    c["steps"] = steps
    return child_text(c)


def assemble(row, idx, planner_every):
    case = {k: row[k] for k in ("id", "question", "document")}
    children = sorted(row["children"], key=lambda c: case["question"].index(c["name"]))
    assert len(children) == 2 and children[0]["unit"] == children[1]["unit"]
    plan = "\n".join(f"Q{i}: {c['name']} | {c['unit']}" for i, c in enumerate(children, 1))
    assert plan_slots(plan) == [(c["name"], c["unit"]) for c in children]
    left, right = children
    if re.search(r"\b(?:less|fewer)\s+than\b", case["question"], re.IGNORECASE):
        left, right = right, left
    amount = int(left["value"]) - int(right["value"])
    if amount != int(row["answer"]):
        raise ValueError(f"Comparison direction mismatch on training row {row['id']}")
    comparison = f"Compare: {left['name']} {left['value']} - {right['name']} {right['value']} = {amount} {left['unit']}"
    final = comparison + f"\nAnswer: {amount}"
    workings = [child_text(c) for c in children]
    joined = "\n".join(
        line for work in workings for line in work.splitlines() if not line.startswith("Value:")
    )
    direct_target = joined + "\n" + final
    examples = []

    def add(stage, target, previous=None):
        examples.append(
            {
                "id": case["id"] + "/" + stage,
                "prompt": make_prompt(stage, case, previous),
                "target": target,
            }
        )

    if idx % planner_every == 0:
        add("plan", plan)
    add("direct", direct_target)
    native = {"plan": {"text": plan}}
    for i, target in enumerate(workings, 1):
        add(f"child{i}", target, native)
    bad = idx % 3 != 0
    slot = (idx // 3) % 2
    variant = (idx // 6) % 4
    noisy = list(workings)
    if bad:
        noisy[slot] = tamper(children[slot], variant, len(case["document"].splitlines()))
    for i, t in enumerate(noisy, 1):
        native[f"child{i}"] = {"text": t}
    bad_draft = "\n".join(
        line for t in noisy for line in t.splitlines() if not line.startswith("Value:")
    )
    if bad and idx % 2:
        bad_draft += f"\nCompare: {right['name']} {right['value']} - {left['name']} {left['value']} = {-amount} {left['unit']}\nAnswer: {-amount}"
    elif bad:
        bad_draft += f"\nAnswer: {amount + 1}"
    else:
        bad_draft = direct_target
    # Include a full corrected trace in supervised review so that the draft can be overruled.
    add("review", direct_target, {"direct": {"text": bad_draft}})
    stage = "full" if (idx // 2 + idx) % 2 == 0 else "compact"
    repair = ""
    if bad:
        repair = "\n".join(workings[slot].splitlines()[:-1]) + "\n"
    add(stage, repair + final, native)
    return examples
