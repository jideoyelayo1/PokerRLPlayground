// Tests of the WebAssembly module (web/dist, from web/build.sh), with Node built-ins only:
// 1. every parity record (weights/demo/<run>.parity.jsonl) through the compiled agent: the same
//    state, mask and chosen action as PyTorch with no_free_fold off and on, Q within 1e-5;
// 2. a seeded replay of HANDS hands with a scripted visitor through the compiled Table, compared
//    line by line with the same replay through the native replay_driver (deals, every move, results);
// 3. the replay's totals, pinned, so that a change of behaviour that both builds share shows;
// 4. through the binding: no view holds an agent's hole cards unless that seat reached showdown,
//    refusals change nothing, and arguments of the wrong JavaScript type are refused, not coerced.
// It also prints the number of decisions per hand, for this visitor and for a calling visitor.
//
// Usage, from the repository root: node web/test.mjs [path/to/replay_driver]

import { spawnSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import createModule from './dist/poker.mjs';

const ROOT = fileURLToPath(new URL('..', import.meta.url));
const driver = process.argv[2] ?? `${ROOT}build/replay_driver`;
const RUN = JSON.parse(readFileSync(`${ROOT}weights/demo/seats.json`, 'utf8')).seats[0];
const SEED = 20261118;
const VISITOR_SEED = 1;
const HANDS = 5000;
const TOLERANCE = 1e-5;

let failures = 0;
const fail = (message) => {
  if (++failures <= 20) console.error(message);
};
const M = await createModule();

// 1. parity records
{
  const lines = readFileSync(`${ROOT}weights/demo/${RUN}.parity.jsonl`, 'utf8').trim().split('\n');
  const f32 = (x) => Math.fround(x);
  let worst = 0;
  let matched = 0;
  for (const [i, line] of lines.entries()) {
    const r = JSON.parse(line);
    const off = M.decideObservation(JSON.stringify(r.observation), false);
    const on = M.decideObservation(JSON.stringify(r.observation), true);
    if (!off.ok || !on.ok) {
      fail(`record ${i}: ${off.error ?? on.error}`);
      continue;
    }
    if (off.state.length !== r.state.length || off.state.some((x, k) => x !== f32(r.state[k]))) fail(`record ${i}: state differs`);
    if (off.mask.some((x, k) => x !== f32(r.mask[k]))) fail(`record ${i}: mask differs`);
    const diff = Math.max(...off.q.map((x, k) => Math.abs(x - f32(r.q[k]))));
    worst = Math.max(worst, diff);
    if (diff > TOLERANCE) fail(`record ${i}: Q differs by ${diff}`);
    if (off.action !== r.action || on.action !== r.action_no_free_fold)
      fail(`record ${i}: actions ${off.action}/${on.action}, PyTorch ${r.action}/${r.action_no_free_fold}`);
    else matched++;
  }
  console.log(`parity: ${lines.length} records; same action as PyTorch, rule off and on, in ${matched}; largest |q - q_torch| ${worst.toPrecision(3)} (tolerance ${TOLERANCE})`);
}

// 2. replay through the WebAssembly Table, in replay_driver's format
function mulberry32(seed) {
  return () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return (t ^ (t >>> 14)) >>> 0;
  };
}

const CARD = /^[2-9TJQKA][hdcs]$/;
const cardsIn = (value, out = new Set()) => {
  if (typeof value === 'string' && CARD.test(value)) out.add(value);
  else if (value && typeof value === 'object') for (const v of Object.values(value)) cardsIn(v, out);
  return out;
};

function replay() {
  const table = new M.Table(SEED);
  const next = mulberry32(VISITOR_SEED);
  const out = [];
  let leaks = 0;
  for (let h = 0; h < HANDS; h++) {
    table.startHand();
    let v = table.view();
    out.push(`hand ${v.hand} button ${v.button} cards ${v.cards.join(' ')}`);
    for (;;) {
      // the only cards a view may hold: the board, the visitor's own and, at the end, the showdown seats'
      const allowed = new Set([...v.board, ...v.cards, ...(v.result?.showdown ?? []).flatMap((s) => s.cards)]);
      for (const c of cardsIn(v)) if (!allowed.has(c)) leaks++;
      for (const s of v.result?.showdown ?? []) if (v.seats[s.seat].status === 'folded') leaks++;
      if (v.over) break;
      const m = v.toAct === v.visitor ? table.act(v.legal[next() % v.legal.length].action) : table.advance();
      if (!m.ok) throw new Error(`refused: ${m.error}`);
      out.push(`${m.seat} ${m.action} ${m.chips} ${m.to}`);
      v = table.view();
    }
    if (!v.result) {
      out.push('end aborted');
      continue;
    }
    const r = v.result;
    out.push(`end board ${v.board.join(' ')} pot ${v.pot} pots ${r.pots.map((p) => `${p.amount}:${p.winners.join(',')}:${p.shares.join(',')}`).join(' ')}` +
      ` net ${r.net.join(' ')} showdown ${r.showdown.map((s) => `${s.seat}:${s.cards.join('')}:${s.hand}`).join(' ')}`);
  }
  table.delete();
  return { out, leaks };
}

const started = Date.now();
const { out: actual, leaks } = replay();
const wasmSeconds = (Date.now() - started) / 1000;
const native = spawnSync(driver, [`${ROOT}weights/demo`, `${SEED}`, `${VISITOR_SEED}`, `${HANDS}`, 'random'], { encoding: 'utf8', maxBuffer: 1 << 28 });
if (native.error || native.status !== 0) {
  console.error(`native driver ${driver} failed:`, native.error?.message ?? native.stderr);
  process.exit(1);
}
const expected = native.stdout.trimEnd().split('\n');
if (expected.length !== actual.length) fail(`native printed ${expected.length} lines, WebAssembly ${actual.length}`);
let firstDiff = -1;
for (let i = 0; i < Math.max(expected.length, actual.length); i++) {
  if (expected[i] !== actual[i]) {
    firstDiff = i;
    fail(`line ${i + 1} differs:\n  wasm   ${actual[i]}\n  native ${expected[i]}`);
    break;
  }
}
if (leaks) fail(`${leaks} views held cards they should not`);

// 3. totals, and decisions per hand
function totals(lines) {
  const t = { hands: 0, aborted: 0, showdowns: 0, actions: {}, visitorActions: {}, net: [0, 0, 0, 0], decisions: [], agentDecisions: [] };
  for (const line of lines) {
    const w = line.split(' ');
    if (w[0] === 'hand') {
      t.hands++;
      t.decisions.push(0);
      t.agentDecisions.push(0);
    } else if (w[0] === 'end') {
      if (w[1] === 'aborted') t.aborted++;
      const net = line.split(' net ')[1]?.split(' showdown')[0].split(' ').map(Number) ?? [0, 0, 0, 0];
      net.forEach((x, s) => (t.net[s] += x));
      if (line.split(' showdown ')[1]) t.showdowns++;
    } else {
      t.actions[w[1]] = (t.actions[w[1]] ?? 0) + 1;
      if (w[0] === '0') t.visitorActions[w[1]] = (t.visitorActions[w[1]] ?? 0) + 1;
      t.decisions[t.decisions.length - 1]++;
      if (w[0] !== '0') t.agentDecisions[t.agentDecisions.length - 1]++;
    }
  }
  return t;
}
function distribution(xs) {
  const s = [...xs].sort((a, b) => a - b);
  const q = (p) => s[Math.min(s.length - 1, Math.floor(p * s.length))];
  const mean = s.reduce((a, b) => a + b, 0) / s.length;
  return `min ${s[0]}, median ${q(0.5)}, mean ${mean.toFixed(1)}, 90th percentile ${q(0.9)}, 99th ${q(0.99)}, max ${s[s.length - 1]}`;
}

const t = totals(actual);
// Fixed by SEED, VISITOR_SEED and HANDS; a behaviour change in the engine, the agent or Table shows here.
const PINNED = {
  hands: 5000, aborted: 0, showdowns: 2635,
  actions: { fold: 11940, check: 8526, call: 17547, raise_bb: 24873, raise_half: 1866, raise_pot: 3064, allin: 3272 },
  visitorActions: { fold: 2163, check: 644, call: 1509, raise_bb: 1903, raise_half: 1617, raise_pot: 1942, allin: 2241 },
  net: [-67335, 34648, 15452, 17235],
};
for (const key of ['hands', 'aborted', 'showdowns'])
  if (t[key] !== PINNED[key]) fail(`${key}: ${t[key]}, pinned ${PINNED[key]}`);
for (const key of ['actions', 'visitorActions'])
  for (const a of new Set([...Object.keys(t[key]), ...Object.keys(PINNED[key])]))
    if (t[key][a] !== PINNED[key][a]) fail(`${key} ${a}: ${t[key][a] ?? 0}, pinned ${PINNED[key][a] ?? 0}`);
if (t.net.join() !== PINNED.net.join()) fail(`chips won per seat: ${t.net.join(' ')}, pinned ${PINNED.net.join(' ')}`);

console.log(`replay: ${HANDS} hands from seed ${SEED}, visitor mulberry32(${VISITOR_SEED}) uniform over its legal actions; ` +
  `${actual.length} lines, ${firstDiff < 0 && expected.length === actual.length ? 'identical to' : 'DIFFERENT from'} the native driver's; ${wasmSeconds.toFixed(1)} s in WebAssembly`);
console.log(`  showdowns ${t.showdowns}, aborted ${t.aborted}; chips won per seat ${t.net.join(' ')}`);
console.log(`  actions, all seats: ${JSON.stringify(t.actions)}`);
console.log(`  actions, visitor: ${JSON.stringify(t.visitorActions)}`);
console.log(`  decisions per hand, all seats: ${distribution(t.decisions)}`);
console.log(`  decisions per hand, agents: ${distribution(t.agentDecisions)}`);
const calling = spawnSync(driver, [`${ROOT}weights/demo`, `${SEED}`, `${VISITOR_SEED}`, `${HANDS}`, 'call'], { encoding: 'utf8', maxBuffer: 1 << 28 });
if (calling.status === 0) {
  const c = totals(calling.stdout.trimEnd().split('\n'));
  console.log(`a calling visitor (native driver, same deals): decisions per hand, all seats: ${distribution(c.decisions)}`);
  console.log(`  agents: ${distribution(c.agentDecisions)}; aborted ${c.aborted}`);
} else fail('native driver, calling visitor, failed');

// 4. the binding: refusals, hidden cards checked above, argument types
{
  const table = new M.Table(7);
  const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
  const expect = (label, got, want) => {
    if (!same(got, want)) fail(`${label}: ${JSON.stringify(got)}, expected ${JSON.stringify(want)}`);
  };
  expect('act before a hand', table.act('call').error, 'hand_over');
  expect('advance before a hand', table.advance().error, 'hand_over');
  let outOfTurn = 0;
  let illegal = 0;
  for (let h = 0; h < 200; h++) {
    table.startHand();
    let v = table.view();
    while (!v.over) {
      if (v.toAct !== v.visitor) {
        expect('act out of turn', table.act('fold').error, 'not_your_turn');
        expect('the view after a refusal', table.view(), v);
        outOfTurn++;
        table.advance();
      } else {
        expect('advance on the visitor\'s turn', table.advance().error, 'visitor_to_act');
        const legal = new Set(v.legal.map((a) => a.action));
        for (const a of M.Table.format().actions.filter((a) => !legal.has(a))) {
          expect(`illegal ${a}`, table.act(a).error, 'illegal_action');
          illegal++;
        }
        for (const bad of [2, null, undefined, true, 2n, new String('call'), ['call'], { action: 'call' }, 'CALL', ''])
          expect(`act(${typeof bad} ${String(bad)})`, table.act(bad).error, 'unknown_action');
        expect('the view after refusals', table.view(), v);
        table.act(v.legal[h % v.legal.length].action);
      }
      v = table.view();
    }
    expect('act after the hand', table.act('fold').error, 'hand_over');
  }
  if (!outOfTurn || !illegal) fail('refusals not exercised');
  table.delete();

  for (const seed of ['5', 1.5, -1, 2 ** 53, NaN, Infinity, 5n, true, null, undefined, {}]) {
    let threw = '';
    try {
      new M.Table(seed).delete();
    } catch (e) {
      threw = e?.constructor?.name;
    }
    if (threw !== 'TypeError') fail(`new Table(${typeof seed} ${String(seed)}) did not throw a TypeError (${threw || 'no exception'})`);
  }
  for (const seed of [0, 2 ** 53 - 1]) new M.Table(seed).delete();
  const o = JSON.parse(readFileSync(`${ROOT}weights/demo/${RUN}.parity.jsonl`, 'utf8').split('\n')[0]).observation;
  const obs = JSON.stringify(o);
  for (const [label, got] of [
    ['decideObservation(object)', M.decideObservation(o, false)],
    ['decideObservation(json, "true")', M.decideObservation(obs, 'true')],
    ['decideObservation(json, 1)', M.decideObservation(obs, 1)],
    ['decideObservation("{}")', M.decideObservation('{}', false)],
    ['decideObservation("[")', M.decideObservation('[', false)],
    ['decideObservation(viewer out of range)', M.decideObservation(JSON.stringify({ ...o, viewer_id: 9 }), false)],
    ['decideObservation(bad card)', M.decideObservation(JSON.stringify({ ...o, hole: [[99, 1], [8, 4]] }), false)],
    ['decideObservation(5 seats)', M.decideObservation(JSON.stringify({ ...o, stack: [...o.stack, 200] }), false)],
  ])
    if (got.ok !== false || typeof got.error !== 'string') fail(`${label}: ${JSON.stringify(got)}, expected a refusal`);
  console.log(`binding: ${outOfTurn} out-of-turn and ${illegal} illegal actions refused with the view unchanged; wrong argument types refused`);
}

if (failures) {
  console.error(`FAIL: ${failures} failure(s)`);
  process.exit(1);
}
console.log('PASS: parity records, replay identical to the native build, totals as pinned, hidden cards and refusals as expected');
