# Compiler accuracy (compile-only track)

Each turn is compiled against the compiler's own committed prior, as in a session. Single run per configuration, temperature 0. Plan-equivalent ignores `task`.

| run | grounding | model | sequences | turns |
|---|---|---|---:|---:|
| compile-ungrounded-20261126T193054Z-612729 | False | ollama/qwen3.5:4b@d8b0f5e9760cd1682034f292d7ef72ec46f432149be0df7574bf2d6e92e38c04 | 12 | 67 |
| compile-grounded-20261126T193508Z-d8e088 | True | ollama/qwen3.5:4b@d8b0f5e9760cd1682034f292d7ef72ec46f432149be0df7574bf2d6e92e38c04 | 12 | 67 |

| kind | n | ungrounded exact | plan | errors | grounded exact | plan | errors |
|---|---:|---:|---:|---:|---:|---:|---:|
| new | 12 | 1.000 | 1.000 | 0 | 1.000 | 1.000 | 0 |
| repeat | 4 | 0.250 | 0.250 | 0 | 0.750 | 0.750 | 0 |
| source_update | 8 | 0.625 | 0.750 | 0 | 0.875 | 0.875 | 0 |
| add_constraint | 10 | 0.100 | 0.100 | 0 | 0.900 | 0.900 | 0 |
| remove_constraint | 2 | 1.000 | 1.000 | 0 | 1.000 | 1.000 | 0 |
| revoke_source | 7 | 0.571 | 0.571 | 0 | 1.000 | 1.000 | 0 |
| add_entity | 5 | 0.600 | 0.600 | 0 | 1.000 | 1.000 | 0 |
| replace_entity | 8 | 0.571 | 0.571 | 1 | 0.875 | 0.875 | 0 |
| add_dimension | 6 | 0.667 | 0.667 | 0 | 1.000 | 1.000 | 0 |
| unrelated | 4 | 1.000 | 1.000 | 0 | 1.000 | 1.000 | 0 |
| remove_dimension | 1 | 0.000 | 0.000 | 0 | 1.000 | 1.000 | 0 |
| ALL | 67 | 0.606 | 0.621 | 1 | 0.940 | 0.940 | 0 |

- ungrounded: model called on 66/67 turns; median compile 3.47 s.

- grounded: model called on 61/67 turns; median compile 3.45 s.
