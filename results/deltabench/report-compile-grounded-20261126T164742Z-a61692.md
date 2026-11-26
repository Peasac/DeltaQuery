# Compiler accuracy (compile-only track)

Each turn is compiled against the compiler's own committed prior, as in a session. Single run per configuration, temperature 0. Plan-equivalent ignores `task`.

| run | grounding | model | sequences | turns |
|---|---|---|---:|---:|
| compile-ungrounded-20261126T163421Z-437ffb | False | ollama/qwen3.5:4b@d8b0f5e9760cd1682034f292d7ef72ec46f432149be0df7574bf2d6e92e38c04 | 40 | 201 |
| compile-grounded-20261126T164742Z-a61692 | True | ollama/qwen3.5:4b@d8b0f5e9760cd1682034f292d7ef72ec46f432149be0df7574bf2d6e92e38c04 | 40 | 201 |

| kind | n | ungrounded exact | plan | errors | grounded exact | plan | errors |
|---|---:|---:|---:|---:|---:|---:|---:|
| new | 40 | 0.975 | 0.975 | 0 | 1.000 | 1.000 | 0 |
| unrelated | 20 | 0.900 | 0.900 | 0 | 1.000 | 1.000 | 0 |
| source_update | 19 | 0.000 | 0.684 | 0 | 1.000 | 1.000 | 0 |
| revoke_source | 25 | 0.000 | 0.600 | 0 | 1.000 | 1.000 | 0 |
| replace_entity | 18 | 0.389 | 0.611 | 0 | 1.000 | 1.000 | 0 |
| add_constraint | 17 | 0.235 | 0.235 | 0 | 1.000 | 1.000 | 0 |
| repeat | 24 | 0.000 | 0.583 | 0 | 1.000 | 1.000 | 0 |
| remove_constraint | 2 | 0.000 | 0.000 | 0 | 1.000 | 1.000 | 0 |
| add_dimension | 21 | 0.857 | 0.857 | 0 | 1.000 | 1.000 | 0 |
| add_entity | 14 | 0.500 | 0.714 | 0 | 1.000 | 1.000 | 0 |
| remove_dimension | 1 | 0.000 | 0.000 | 0 | 1.000 | 1.000 | 0 |
| ALL | 201 | 0.463 | 0.706 | 0 | 1.000 | 1.000 | 0 |

- ungrounded: model called on 201/201 turns; median compile 3.40 s.

- grounded: model called on 133/201 turns; median compile 3.36 s.
