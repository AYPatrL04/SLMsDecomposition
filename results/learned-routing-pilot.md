# Learned routes and source-checked handoffs

This archived feasibility study holds each proposed route and its raw child outputs fixed. The repair condition replaces only a value that the cited evidence determines. Invalid routes, missing outputs and capped calls remain in the assigned denominator.

| Panel and model | Usable model routes or programs | Raw handoff | Verified handoff | Controls |
| --- | ---: | ---: | ---: | --- |
| 36 fresh signed counting problems, Qwen3.5-4B | 14/36 schema-valid, 0/36 complete paths | 0/36 | 0/36 | Direct 0/36, generated expression 1/36, full-source symbolic 36/36 |
| Same 36 problems, SmolLM3-3B | 0/36 | 0/36 | 0/36 | Direct and generated expression 0/36, full-source symbolic 36/36 |
| 12 fresh counting problems, Qwen planner with the study adapter for the child calls | 0/12 | 0/12 | 0/12 | Adapted Direct 4/12, supplied route 10/12 with either handoff, full-source symbolic 12/12 |
| 24 FinQA development questions, Qwen3.5-4B | 2/24 executable | 0/24 | 0/24 | Direct 4/24, exact execution of the same model program 0/24 |
| 8 other FinQA development questions, shorter program format | 5/8 executable | 0/8 | 0/8 | Direct and exact execution of the same model program 0/8 |

With a supplied route on the 12-problem counting panel, the adapted child solver returned 24/24 correct endpoints. The neural merger lost the sign on two negative answers. Applying the requested signed subtraction to the same endpoints solved 12/12. All 24 single-operand planner calls reached their 1,536-token cap before returning a route.

The [FinQA](https://aclanthology.org/2021.emnlp-main.300/) panels use different development filings. These runs examined executability and arithmetic grounding. Several executable programs selected the wrong financial quantity or operation. The official test split was held out.

A separate audit used the 2,048 archived terminal operand records from the main counting study. The checker accepted 2,015 supported values, identified 18 contradicted values with correct alternatives and abstained on 15 insufficient records. Perturbing each originally correct supported value by one produced 2,015 correct replacements. Omitting or conflicting with the cited anchor produced 2,015 abstentions in each condition. These are local state outcomes on the signed-additive grammar.

The experiment supports selective local repair under sufficient citations and leaves its end-to-end benefit under model-generated routes unresolved. This summary contains aggregates from the archived experiments. The CLI in this repository runs the fixed MathGAP study.
