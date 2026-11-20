# Fixed MathGAP splits

These gzip-compressed JSON files let a checkout repeat training and evaluation on the precise questions used in the study. They derive from [MathGAP revision 5724e4f](https://github.com/eth-lre/mathgap-experiments/tree/5724e4f65048a62c87da8705c73117a04a4457c3). The upstream license is included in this directory.

| File | Rows | Contents | SHA-256 after decompression |
|---|---:|---|---|
| train.json.gz | 512 | Short training parents with annotated operand traces | `3613e53da9cb4bd20b25a1ae1a598a6fac18f13a61b3d07c3b9ac560a43d710c` |
| inputs.json.gz | 1,024 | Opaque ID, public English document and question, in evaluation order | `8a81ca801df0c52b19d7e9d347fd9a750c74b85c9b7af2705c0c0513669dc52d` |
| references.json.gz | 1,024 | ID-keyed answer, comparison direction, proof-family ID and gold intermediate and terminal quantities for scoring | `5de5ae153956571aab544e337a2aa7440c42ad9018aa01957a42b252074fd445` |

The original allocator screened candidate evaluation problems against 1,184 previously allocated questions, and balanced “more” and “fewer” wording (512 each). All 1,024 have distinct pairs of signed eight-update sequences and distinct multisets of premise numbers. The vocabulary, grammar and two-chain topology are shared across splits. The fixed files remove the need to reconstruct the historical allocation or install a second generator environment.

The inference command reads only the public inputs. Scoring opens the reduced reference file after checking the completion receipt. Training constructs 2,688 prompt/target examples from its separate annotated file. Their SHA-256 in original training order, using sorted-key JSON serialization, is `05f28ccef3e26d378c02643d5419f8b2ea864a50f054e197abc28591427a6f9e`.
