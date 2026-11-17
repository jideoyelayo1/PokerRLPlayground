# State encoding, action indices and legal mask

This is what `encode_observation(obs, big_blind)` and `legal_mask_from_obs(obs)` in
`ReinforcementLearningEnvironment/helper.py` compute, written so that it can be re-implemented
(in C++, for example) without reading the Python. The input is the engine's `Observation` for
the player about to act (`GameEngine::getObservation(viewer)`); `n` is the number of seats
(`obs.stack.size()`), 4 for the demo agents, and `v = obs.viewer_id`.

## Vector

`127 + 6n` float32 values (151 for 4 seats), in this order.

| index (n = 4) | count | element | value |
|---|---:|---|---|
| 0–16 | 17 | hole card 1, `obs.hole[0]` | card one-hot (below) |
| 17–33 | 17 | hole card 2, `obs.hole[1]` | card one-hot |
| 34–118 | 5 × 17 | board cards `obs.board[0..4]` | card one-hot each; a card not dealt yet is all zeros |
| 119 | 1 | pot | `obs.pot_total` × c |
| 120 | 1 | highest street contribution | `obs.to_call` × c |
| 121 | 1 | minimum raise increment | `obs.min_raise` × c |
| 122 | 1 | amount the viewer needs to call | `obs.viewer_to_call` × c |
| 123–126 | 4 | street | one-hot of `obs.street` (0 preflop, 1 flop, 2 turn, 3 river); all zeros if `obs.street` is outside 0–3 |
| 127–146 | 5n | seats, relative to the viewer | for k = 0 … n−1, seat `i = (v + k) mod n`: `stack[i]` × c, `folded[i]` (0/1), `all_in[i]` (0/1), `street_contrib[i]` × c, `total_contrib[i]` × c |
| 147–150 | n | button, relative to the viewer | one-hot at position `(obs.button − v) mod n` (0 means the viewer has the button); all zeros if `obs.button < 0` |

So k = 0 is always the viewer and k = 1 the seat after it, and the same network can play any
seat. With 4 seats and the button relative position `b`: `b = 0` button, `b = 3` small blind,
`b = 2` big blind, `b = 1` first to act preflop.

**Chip scale.** `c = 1 / (100 × big_blind)`: chip amounts are in units of 100 big blinds, the
demo's starting stack (200 chips at blinds 1/2, so `c = 1/200`). For bit-identical results,
compute `c` in double precision, multiply the integer chip count by it in double precision, then
round to float32 (`(float)(chips * (1.0 / (100.0 * big_blind)))`). The 0/1 flags and one-hots are
exact.

**Card one-hot.** 17 values: 13 for the rank, then 4 for the suit. Rank `r` (2 … 14, ace = 14)
sets element `r − 2`; suit `s` (1 hearts, 2 diamonds, 3 clubs, 4 spades) sets element `13 + s − 1`.
An empty card (`rank == 0`, `suit == 0`) is all zeros. The engine hides a folded viewer's hole
cards (all zeros), but a folded player is never asked to act.

## Action indices

| index | action | engine `Action` |
|---:|---|---|
| 0 | fold | `{Fold}` |
| 1 | check | `{Check}` |
| 2 | call | `{Call}` |
| 3 | raise to current bet + 1 big blind | `{Raise, RaiseSize::BB}` |
| 4 | raise to current bet + half the pot | `{Raise, RaiseSize::HalfPot}` |
| 5 | raise to current bet + the pot | `{Raise, RaiseSize::Pot}` |
| 6 | all-in | `{AllIn}` |

## Legal mask

`obs.legal` (from `legalActions`) has one `ActionSpec` per action above: Fold, Check, Call,
AllIn, then one Raise per `RaiseSize` with `raise_size` and `raise_to` set. Each `allowed` flag is
exactly whether `step()` accepts that action. The mask is a float32 vector of 7: 0 where the
action's spec is allowed, −1e9 where it is not. A network's choice is
`argmax(q + mask)`, ties going to the lowest index (PyTorch's `argmax` returns the first maximum).
