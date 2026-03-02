# Shakespeare Character-Level Transformer

A minimal example that trains a small GPT-style transformer to generate Shakespeare-like text. Runs in **~3 minutes on an L4 GPU** and costs about $0.02.

## What it does

- Downloads the [Tiny Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) dataset (~1MB, 65 character vocabulary)
- Trains a 6-layer, 192-dim transformer (~2.7M parameters)
- Logs `train/loss` and `eval/loss` to `bp/metrics.jsonl`
- Saves text samples at intervals and a final 1000-char generation
- Saves the model checkpoint to `bp/outputs/model.pt`

## Run with Backprop

```bash
bp run train.py --gpu l4
```

That's it. Backprop will snapshot the code, spin up a GPU, install dependencies, and stream logs back to you. Watch the loss curves in the web UI at [backprophq.com](https://backprophq.com).

## Run locally (for testing)

Run from the project root:

```bash
uv run train.py
```

`uv` handles the virtualenv and dependencies automatically. Works on CPU too (just much slower — ~15 minutes).

## Tuning

All hyperparameters are at the top of `train.py`. Some things to try:

| Change                      | Effect                              |
| --------------------------- | ----------------------------------- |
| `MAX_STEPS = 10000`         | Lower loss, better samples (~6 min) |
| `N_LAYER = 4, N_EMBD = 128` | Faster training, slightly worse     |
| `BLOCK_SIZE = 512`          | More context, more VRAM             |
| `gpu = "a100"`              | ~2x faster, useful for bigger runs  |

## Example output

After ~5000 steps you get output like:

```
ROMEO:
What say'st thou? I do not know the world,
That thou hast spoke of in the name of death;
For I have seen the day of such a sight
And made the morning with a thousand tears.
```

Not exactly Shakespeare, but recognizably theatrical — from a model that trained for 3 minutes.
