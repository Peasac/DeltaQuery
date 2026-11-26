# Compiler accuracy (compile-only track)

Each turn is compiled against the compiler's own committed prior, as in a session. Single run per configuration, temperature 0. Plan-equivalent ignores `task`.

| run | grounding | model | sequences | turns |
|---|---|---|---:|---:|
| compile-ungrounded-20261126T105346Z-7db10f | False | ollama/qwen3.5:4b@d8b0f5e9760cd1682034f292d7ef72ec46f432149be0df7574bf2d6e92e38c04 | 40 | 206 |
| compile-grounded-20261126T110749Z-2b2c62 | True | ollama/qwen3.5:4b@d8b0f5e9760cd1682034f292d7ef72ec46f432149be0df7574bf2d6e92e38c04 | 40 | 206 |

| kind | n | ungrounded exact | plan | errors | grounded exact | plan | errors |
|---|---:|---:|---:|---:|---:|---:|---:|
| new | 40 | 0.975 | 0.975 | 0 | 0.975 | 0.975 | 0 |
| source_update | 25 | 0.000 | 0.600 | 0 | 0.960 | 0.960 | 0 |
| repeat | 20 | 0.000 | 0.600 | 0 | 0.950 | 0.950 | 0 |
| add_constraint | 17 | 0.118 | 0.118 | 0 | 0.941 | 0.941 | 0 |
| replace_entity | 18 | 0.500 | 0.833 | 0 | 1.000 | 1.000 | 0 |
| revoke_source | 23 | 0.000 | 0.652 | 0 | 0.913 | 0.913 | 0 |
| add_dimension | 28 | 0.714 | 0.714 | 0 | 0.964 | 0.964 | 0 |
| unrelated | 11 | 0.909 | 0.909 | 0 | 0.909 | 0.909 | 0 |
| add_entity | 14 | 0.357 | 0.786 | 0 | 1.000 | 1.000 | 0 |
| remove_dimension | 6 | 0.667 | 0.833 | 0 | 1.000 | 1.000 | 0 |
| remove_constraint | 2 | 0.000 | 0.000 | 0 | 1.000 | 1.000 | 0 |
| remove_entity | 2 | 0.000 | 1.000 | 0 | 1.000 | 1.000 | 0 |
| ALL | 206 | 0.432 | 0.709 | 0 | 0.961 | 0.961 | 0 |

- ungrounded: model called on 206/206 turns; median compile 3.47 s.

- grounded: model called on 138/206 turns; median compile 3.31 s.
