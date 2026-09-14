# Design and findings

## From scene to code

`backend/core/scene.py` generates numerical comparison scenes. Agent variants encode a sequence, write to a bounded scratch space, and read that representation to answer tasks. `backend/evaluation` collects predictions and examines code structure, task accuracy, and generalization.

Limited memory and quantization are experimental constraints. Good task accuracy alone does not show that a model has invented a compositional symbol system: inspect whether different quantities have distinct codes and whether the code is actually used by the prediction path.

## A failed architecture

The selected V32 records describe a model that remained near the chance loss and often wrote a constant code. `scripts/v32_init_probe.py` inspects where quantity information is lost between layers. The saved report records larger activation spread after initialization changes, followed by continued training failure.

Those observations are specific to that architecture and its recorded runs. They are not current performance claims for every model retained in this repository. The [historical findings](../research/v32_findings/README.md) identify the attempts and original records.

## Label-mapping example

Training and evaluation must agree on what each class index means. For a balanced signed-offset task with `k_max=2`, offsets outside the fine range get separate far-negative and far-positive classes. Equal quantities have their own middle class.

Run from the repository root:

```sh
python -c "from backend.core.label_mapping import k_to_class; print([k_to_class(k, 2) for k in (-9, -2, -1, 0, 1, 2, 9)])"
```

Expected output:

```text
[0, 1, 2, 3, 4, 5, 6]
```

`eval_k_to_class` selects the balanced or legacy mapping from the scene configuration. The integration regression test checks that training and evaluation use the same mapping. Running this small example validates only the mapping, not a model's predictions or training quality.
