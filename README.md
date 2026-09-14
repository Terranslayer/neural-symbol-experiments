# Neural Symbol Experiments

Independent Python and PyTorch experiments on how neural agents encode quantities in limited memory and use those representations for comparison tasks.

This focused copy of my SymbolicAI work contains model implementations, scene generators, training and evaluation code, diagnostics, and tests. It complements [SymbolEmergence](https://github.com/Terranslayer/learned-number-representations), which studies transfer to a newly trained reader.

## Questions explored

- Does a model preserve quantity information as it passes through the network?
- Does quantization produce distinct codes or collapse different inputs to the same code?
- Do training and evaluation assign the same labels to a numerical relationship?

## Two useful examples

**Representation collapse.** In the saved V32 diagnostic run, most agents produced the same code for different quantities. Layer-by-layer probes inspect activations before and after the write head to help locate that loss of information. Improving initial activation spread did not establish that training learned a useful symbol system.

**Evaluation consistency.** A balanced-label training preset and an older evaluation mapping assigned different classes to the same offsets. `backend/core/label_mapping.py` now supplies the mapping to both paths, with regression coverage in `backend/tests/test_v35_eval_label_consistency.py`.

See [design and findings](docs/design.md) for the source paths and limitations. This is exploratory research, not a claim of general symbolic reasoning.

## Code map

```text
backend/core/          Agents, scenes, memory, and shared label mapping
backend/training/      Training loops
backend/evaluation/    Collection, transfer, and representation metrics
backend/tests/         Model and evaluation regression checks
scripts/              Inspection and diagnostic commands
research/v32_findings/ Selected historical diagnostic records
```

`backend/core/agent.py` is a smaller starting point than the later experimental architectures. For a focused debugging example, read `label_mapping.py` alongside its regression test.

## Environment and scope

The CUDA experiments used Linux/WSL and PyTorch; some variants depend on Mamba and compiled CUDA extensions. `backend/requirements.txt` records an older environment and its setup notes. It is not a verified install recipe for every later branch or current machine.

Source and selected saved records are included. Training checkpoints, the separate visualization frontend, later unrelated research branches, and personal run-management files are not included. The code has not been freshly trained or fully validated as a standalone GPU environment in this snapshot.

The shared label-mapping module itself uses only Python's standard library. The [small example](docs/design.md#label-mapping-example) can be run without installing the ML stack.
