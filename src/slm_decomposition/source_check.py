"""Validate cited English comparison facts. This module never reads files or model labels."""

import re
from collections import defaultdict, deque
from dataclasses import dataclass

NAME = r"[A-Za-z][A-Za-z'’\-]*"
UNIT = r"[A-Za-z][A-Za-z'’\-]*(?: [A-Za-z][A-Za-z'’\-]*)*?"
NUMBER = r"[+-]?\d{1,9}"
FACT = re.compile(
    rf"({NAME}) has ({NUMBER}) ({UNIT})(?: (more|fewer|less) than ({NAME}))?\.",
    re.IGNORECASE,
)
QUESTION = re.compile(
    rf"How many ({UNIT}) does ({NAME}) have (more|fewer|less) than ({NAME})\?",
    re.IGNORECASE,
)
CITED = re.compile(r"^\s*\[?S([1-9]\d{0,2})\]?\s*:\s*(.*?)\s*$", re.IGNORECASE)
BASE = re.compile(rf"({NAME})\s*=\s*({NUMBER})\s+({UNIT})[.]?", re.IGNORECASE)
STEP = re.compile(
    rf"({NAME})\s*=\s*({NAME})\s+({NUMBER})\s*([+-])\s*({NUMBER})\s*=\s*({NUMBER})\s+({UNIT})[.]?",
    re.IGNORECASE,
)
VALUE = re.compile(rf"Value\s*:\s*({NAME})\s*=\s*({NUMBER})\s+({UNIT})[.]?", re.IGNORECASE)
ANSWER = re.compile(rf"Answer\s*:\s*({NUMBER})[.]?", re.IGNORECASE)


@dataclass(frozen=True)
class Fact:
    subject: str
    unit: str
    amount: int
    other: str | None


def parse_fact(line):
    match = FACT.fullmatch(line.strip())
    if not match:
        return None
    subject, amount, unit, relation, other = match.groups()
    sign = -1 if relation and relation.lower() in {"less", "fewer"} else 1
    return Fact(
        subject.casefold(), unit.casefold(), sign * int(amount), other.casefold() if other else None
    )


def query(case):
    match = QUESTION.fullmatch(case["question"].strip())
    if not match:
        return None
    unit, left, relation, right = match.groups()
    return (
        unit.casefold(),
        left.casefold(),
        right.casefold(),
        1 if relation.casefold() == "more" else -1,
    )


def compose_values(case, values):
    """Compose keyed counts in the direction specified explicitly by the question."""
    parsed = query(case)
    if not parsed:
        return None
    _, left, right, sign = parsed
    owned = {name.casefold(): n for name, n in values.items() if type(n) is int}
    if left not in owned or right not in owned:
        return None
    return sign * (owned[left] - owned[right])


def resolve(facts):
    """Reconstruct anchored quantities within each unit; abstain on a contradiction."""
    graph = defaultdict(list)
    queue = deque()
    for fact in facts:
        subject = (fact.subject, fact.unit)
        if fact.other is None:
            queue.append((subject, fact.amount))
        else:
            other = (fact.other, fact.unit)
            graph[other].append((subject, fact.amount))
            graph[subject].append((other, -fact.amount))
    known = {}
    while queue:
        key, value = queue.popleft()
        if key in known:
            if value != known[key]:
                return {}
            continue
        known[key] = value
        queue.extend((target, value + delta) for target, delta in graph[key])
    return known


def solve_source(case):
    """Task-specific symbolic baseline with access to all sentences in the raw question."""
    parsed = query(case)
    facts = [parse_fact(line) for line in case["document"].splitlines() if line.strip()]
    if not parsed or any(fact is None for fact in facts):
        return None
    known = resolve(facts)
    return compose_values(case, {who: n for (who, unit), n in known.items() if unit == parsed[0]})


def valid_line(fact, content, known):
    if fact.other is None:
        match = BASE.fullmatch(content)
        return bool(
            match
            and match[1].casefold() == fact.subject
            and int(match[2]) == fact.amount
            and match[3].casefold() == fact.unit
        )
    match = STEP.fullmatch(content)
    if not match:
        return False
    owner, other, old, op, magnitude, result, unit = match.groups()
    owner, other, unit = owner.casefold(), other.casefold(), unit.casefold()
    delta = int(magnitude) * (1 if op == "+" else -1)
    right_edge = (owner == fact.subject and other == fact.other and delta == fact.amount) or (
        other == fact.subject and owner == fact.other and delta == -fact.amount
    )
    return (
        unit == fact.unit
        and right_edge
        and known.get((other, unit)) == int(old)
        and int(result) == int(old) + delta
        and known.get((owner, unit)) == int(result)
    )


def inspect_work(case, text, owner=None, unit=None):
    """Use only source rows cited by generated step lines, then compare the terminal claim."""
    source = case["document"].splitlines()
    parsed = query(case)
    if unit is None and parsed:
        unit = parsed[0]
    unit = unit.casefold() if unit else None
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    citations = []
    bad_ids = []
    for line in lines:
        mark = CITED.fullmatch(line)
        if not mark:
            continue
        idx = int(mark[1])
        fact = parse_fact(source[idx - 1]) if idx <= len(source) else None
        if fact is None:
            bad_ids.append(idx)
        else:
            citations.append((idx, fact, mark[2]))
    known = resolve([fact for _, fact, _ in citations])
    failing = [idx for idx, fact, content in citations if not valid_line(fact, content, known)]
    if owner:
        supported = known.get((owner.casefold(), unit))
        last = VALUE.fullmatch(lines[-1]) if lines else None
        reported = (
            int(last[2])
            if last and owner.casefold() == last[1].casefold() and unit == last[3].casefold()
            else None
        )
        label = f"{owner} ({unit})"
    else:
        supported = compose_values(case, {who: n for (who, u), n in known.items() if u == unit})
        last = ANSWER.fullmatch(lines[-1]) if lines else None
        reported = int(last[1]) if last else None
        label = "requested comparison"
    feedback = []
    if failing or bad_ids:
        labels = ", ".join(f"S{i}" for i in sorted(set(failing + bad_ids)))
        feedback.append(f"Recheck the cited direction, operands or statement at {labels}.")
    if supported is None:
        status = "unsupported"
        feedback.append(f"The cited statements do not establish {label}.")
    else:
        status = (
            "verified" if reported == supported and not failing and not bad_ids else "corrected"
        )
        feedback.append(f"Cited evidence supports {label} = {supported}.")
    return {
        "status": status,
        "feedback": " ".join(feedback),
        "supported_value": supported,
        "corrected_value": supported if status == "corrected" else None,
        "reported_value": reported,
        "verified_steps": len(citations) - len(failing),
        "invalid_steps": len(failing) + len(bad_ids),
        "cited_ids": sorted({i for i, _, _ in citations}),
    }
