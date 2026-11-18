# Trained agents

Trained by `bench/train_demo.py` under the rules in [`bench/SELECTION.md`](../bench/SELECTION.md),
evaluated by `bench/eval_demo.py` (results in [`bench/RESULTS.md`](../bench/RESULTS.md), section 6),
exported by `bench/export_weights.py`.

## Selected

All three agent seats are played by one network, **`mlp-s-seed3`**: an MLP 151 → 128 → 128 → 7 with
ReLU (36,871 parameters; `demo/mlp-s-seed3.bin` is 147,484 bytes, sha256
`c6d70e3c0b18b79c72218d5a1f814a782562552effc6dca1b40c95504aafa8d4`), the checkpoint at 320,000
of its 400,000 training hands. The selection rule put it in all three seats because it was clearly
best (`bench/SELECTION.md`, `demo/seats.json`). It is an MLP, so it needs no history.

Provenance: every run's `run.json` gives as `commit` the head of the branch when the run ended
(other runs' weights and documentation were committed while it trained); the training code is the
same in abdf6f8, where the runs started, and in every commit recorded. From 1b0446f on,
`train_demo.py` records the commit a run starts from.

## Layout

- `runs/<candidate>-seed<seed>/`, one folder per training run, selected or not:
  - `model.pt`: the state_dict of the checkpoint the run kept (best on the validation deals);
  - `log.jsonl`: one line per checkpoint: hands trained, wall and CPU seconds, epsilon, mean loss,
    and the validation result against each baseline (bb/100 and 95% interval, actions chosen);
  - `run.json`: candidate, architecture, seed, hyperparameters, commit, timings, kept checkpoint.
- `demo/`, the selected agents, for the browser demo and the C++ forward pass:
  - `seats.json`: which run plays each of the three agent seats, with sizes and checksums;
  - `<run>.pt`: PyTorch state_dict, `torch.load(path, weights_only=True)`;
  - `<run>.bin`: the same tensors as little-endian float32, C order (row-major), concatenated in
    state_dict order with no padding;
  - `<run>.json`: manifest: `tensors` (name, shape, byte offset, byte count), `sha256` of the
    `.bin`, `architecture` and `forward` (the forward pass in words), the training seed, commit
    and hyperparameters;
  - `<run>.parity.jsonl`: decision points from evaluation play, for checking a re-implementation.

`python -m unittest discover -s tests` checks that every `.bin` holds the same numbers as its `.pt`,
matches its checksum and is at most 8 MB, and that every parity record reproduces.

## Playing one decision

Inputs, at a decision of the agent's seat `v`:
1. `obs = engine.getObservation(v)`;
2. `x = encode_observation(obs, big_blind=2)`: 151 float32 values, specified in
   [`docs/state_encoding.md`](../docs/state_encoding.md);
3. `mask = legal_mask_from_obs(obs)`: 7 values, 0 for a legal action and −1e9 for an illegal one.

Network input: an MLP gets `x` alone, shape `[1, 151]`. A GRU gets the sequence of the seat's
vectors at each of its own decisions in this hand, the current one last, shape `[1, k, 151]` at
its k-th decision; the sequence is emptied at the start of every hand. (Equivalently, keep the
GRU's hidden state between the seat's decisions and reset it to zero each hand.)

Output: 7 Q-values in units of 100 big blinds. The action is `argmax(q + mask)` (first index on a
tie): 0 fold, 1 check, 2 call, 3 raise to the current bet + 1 big blind, 4 raise to the current bet
+ half the pot, 5 raise to the current bet + the pot, 6 all-in. Play it with
`engine.step(action_index_to_engine_action(a), v)` (the table in `docs/state_encoding.md`); with
the exact legal mask the engine always accepts it.

## Parity records

One JSON object per line:

| field | content |
|---|---|
| `baseline` | the baseline the agent was playing (three copies of it) |
| `hand` | hand number within that baseline's evaluation play |
| `observation` | the raw `Observation` at the decision: every field, cards as `[rank, suit]` (rank 2–14, suit 1 hearts, 2 diamonds, 3 clubs, 4 spades; `[0, 0]` empty), `legal` as a list of `{type, raise_size, raise_to, call_amount, min_raise_to, max_raise_to, allowed}` with `type` 0 fold, 1 check, 2 call, 3 raise, 4 all-in and `raise_size` 0 big blind, 1 half pot, 2 pot |
| `state` | `encode_observation(observation)`, 151 float32 values |
| `history` | GRU only: the earlier decisions of the same seat in the same hand, oldest first, each `{observation, state}`; the network input is the `history` states followed by `state`. Empty for an MLP |
| `mask` | the legal mask (7 values) |
| `q` | the PyTorch network's output (7 float32 values) |
| `action` | the action index played, `argmax(q + mask)` |

Every float is printed so that it parses back to the same float32. Recomputing `q` with another
BLAS or another implementation of the same arithmetic can differ in the last bits; the test
allows 1e-5.
