"""Prompt contract for source-grounded MathGAP composition, shared by training and inference."""

import re

from .source_check import inspect_work

INT = r"[+-]?\d+"
PERSON = r"[A-Za-z][A-Za-z'’-]*"
UNIT = r"[A-Za-z]+(?: [A-Za-z]+)*"
SLOTS = re.compile(rf"^Q([12]):\s*({PERSON})\s*\|\s*({UNIT})\s*$", re.IGNORECASE | re.MULTILINE)
VALUE = re.compile(rf"Value:\s*({PERSON})\s*=\s*({INT})\s+({UNIT})", re.IGNORECASE)
ANSWER = re.compile(rf"Answer:\s*({INT})", re.IGNORECASE)
TRACE = (
    "Trace each original quantity to its numbered source. Use one line per fact. "
    "For a stated amount use 'S1: Ava = 12 apples'. For a comparison use "
    "'S2: Mia = Ava 12 - 3 = 9 apples'. The sentence number is the source of that step. "
    "Use + or - in the direction from the known owner to the owner being derived. "
    "These examples illustrate format only."
)
FINISH = (
    "For the final comparison, show 'Compare: <owner> <amount> - <owner> <amount> = "
    "<result> <unit>', choosing the operand order requested by the question. "
    "When the question asks how many fewer A has than B, subtract A's amount from B's. "
    "End on 'Answer: <integer>'."
)


def last_line(text):
    return next((s.strip() for s in reversed(text.splitlines()) if s.strip()), "")


def plan_slots(text):
    slots = {}
    for n, name, unit in SLOTS.findall(text):
        index = int(n)
        slots[index] = None if index in slots else (name, unit.strip())
    return [slots.get(i) or (None, None) for i in (1, 2)]


def answer(text):
    m = ANSWER.fullmatch(last_line(text))
    return int(m[1]) if m else None


def child_value(text):
    m = VALUE.fullmatch(last_line(text))
    return (m[1], int(m[2]), m[3]) if m else None


def make_prompt(stage, case, previous=None, *, feedback=True):
    previous = previous or {}
    document = "\n".join(f"S{i}: {line}" for i, line in enumerate(case["document"].splitlines(), 1))
    head = f"Source statements:\n{document}\nQuestion: {case['question']}\n\n"
    if stage == "plan":
        return head + (
            "Identify the first and second person as mentioned in the question, and the unit. "
            "Request their individual quantities. Write exactly two lines, "
            "'Q1: <first person> | <unit>' and 'Q2: <second person> | <unit>'."
        )
    if stage == "direct":
        return head + f"Solve from the original statements. {TRACE} {FINISH}"
    if stage == "review":
        draft = previous["direct"]["text"]
        findings = f"Source check: {inspect_work(case, draft)['feedback']}\n" if feedback else ""
        return (
            head + f"Whole solution to review:\n{draft}\n{findings}"
            f"Verify the two people, their amounts and comparison direction. Correct any errors. "
            f"Use the source if the draft is incomplete. {TRACE} {FINISH}"
        )
    slots = plan_slots(previous["plan"]["text"])
    if stage in ("child1", "child2"):
        owner, unit = slots[int(stage[-1]) - 1]
        if not owner:
            return None
        return head + (
            f"Subproblem: How many {unit} does {owner} have? Work only on this person's amount. "
            f"{TRACE} End on 'Value: {owner} = <integer> {unit}'."
        )
    if stage not in ("full", "compact"):
        raise ValueError(stage)
    notes = []
    for i, (owner, unit) in enumerate(slots, 1):
        result = previous[f"child{i}"]["text"]
        if not owner:
            notes.append(f"Operand {i}: missing assigned person.")
            continue
        if stage == "full":
            given = result
        else:
            value = child_value(result)
            given = f"Value: {value[0]} = {value[1]} {value[2]}" if value else "Value: missing"
        note = f"Operand {i} ({owner}, {unit}):\n{given}"
        if feedback:
            findings = inspect_work(case, result, owner, unit)
            note += f"\nSource check: {findings['feedback']}"
        notes.append(note)
    return (
        head
        + "\n".join(notes)
        + (
            f"\nCompose a final solution. Retain supported amounts. Repair unsupported "
            f"amounts from the original statements. {FINISH}"
        )
    )
