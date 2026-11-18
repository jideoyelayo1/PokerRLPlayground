// Tests of the WebAssembly module (web/dist, from web/build.sh), with Node built-ins only. Sections
// 2-4 run against the release module itself; section 1 needs decideObservation, which only the test
// module (build/wasm-test/poker_test.mjs: the release sources plus web/test_hooks.cpp) has.
// 1. every parity record (weights/demo/<run>.parity.jsonl) through the compiled agent: the same
//    state, mask and chosen action as PyTorch with no_free_fold off and on, Q within 1e-5, and Q
//    bit-identical to the native build's (replay_driver q). This shows parity on these records; it does
//    not guarantee it: in play the top two Q-values can be closer than the C++-PyTorch difference;
// 2. a seeded replay of HANDS hands with a scripted visitor through the compiled Table, compared
//    line by line with the same replay through the native replay_driver (deals, every move, results);
// 3. the replay's totals, pinned, so that a change of behaviour that both builds share shows;
// 4. through the binding: no view holds an agent's hole cards unless that seat reached showdown,
//    refusals change nothing, and arguments of the wrong JavaScript type are refused, not coerced;
// 5. out of memory: the constructor throws a RangeError, and the module recovers once Tables are deleted.
// It also prints the number of decisions per hand, for this visitor and for a calling visitor.
//
// Usage, from the repository root: node web/test.mjs [path/to/replay_driver]

import { spawnSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import createModule from './dist/poker.mjs';
import createTestModule from '../build/wasm-test/poker_test.mjs';

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
const M = await createModule(); // the release module
const T = await createTestModule(); // the test module
for (const hook of ['decideObservation', 'stackPointer', 'playUnchecked'])
  if (hook in M) fail(`the release module must not have the test hook ${hook}`);

// 1. parity records
{
  const lines = readFileSync(`${ROOT}weights/demo/${RUN}.parity.jsonl`, 'utf8').trim().split('\n');
  const f32 = (x) => Math.fround(x);
  let worst = 0;
  let matched = 0;
  for (const [i, line] of lines.entries()) {
    const r = JSON.parse(line);
    const off = T.decideObservation(JSON.stringify(r.observation), false);
    const on = T.decideObservation(JSON.stringify(r.observation), true);
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
  // the native build's Q-values, bit for bit (both are built with -ffp-contract=off)
  const bits = (x) => {
    const b = new DataView(new ArrayBuffer(4));
    b.setFloat32(0, x);
    return b.getUint32(0).toString(16).padStart(8, '0');
  };
  const nativeQ = spawnSync(driver, [`${ROOT}weights/demo`, 'q'], { encoding: 'utf8', maxBuffer: 1 << 28 }).stdout.trimEnd().split('\n');
  let identical = 0;
  for (const [i, line] of lines.entries())
    identical += T.decideObservation(JSON.stringify(JSON.parse(line).observation), false).q.map(bits).join(' ') === nativeQ[i];
  if (identical !== lines.length) fail(`Q-values bit-identical to the native build's in ${identical} of ${lines.length} records`);
  console.log(`parity: ${lines.length} records; same action as PyTorch, rule off and on, in ${matched}; largest |q - q_torch| ${worst.toPrecision(3)} (tolerance ${TOLERANCE}); ` +
    `Q-values bit-identical to the native build's in ${identical}`);
}

// 0. refused calls must not leak WebAssembly stack. When the C++ side threw the seed's TypeError,
// each refused new Table() left 16 bytes on the shadow stack and the module failed after 4,499.
// 100,000 refused calls of each kind on both modules: on the test module the stack pointer must not
// have moved; on the release module the replay below, run afterwards, must still be identical.
function refuseMany(Mod) {
  const N = 100000;
  const BAD = ['5', 1.5, -1, 2 ** 53, NaN, Infinity, 5n, true, null, undefined, {}];
  const counts = { construct: 0, beforeHand: 0, inHand: 0, parse: 0 };
  for (let i = 0; i < N; i++) {
    try {
      new Mod.Table(BAD[i % BAD.length]).delete();
    } catch (e) {
      counts.construct += e instanceof TypeError;
    }
  }
  const t = new Mod.Table(1);
  for (let i = 0; i < N; i++) counts.beforeHand += t.act('fold').error === 'hand_over' && t.advance().error === 'hand_over';
  for (let v = t.view(); v.toAct !== v.visitor; v = t.view()) v.over ? t.startHand() : t.advance();
  const illegal = Mod.Table.format().actions.find((a) => !t.view().legal.some((l) => l.action === a));
  for (let i = 0; i < N; i++)
    counts.inHand += t.startHand().error === 'hand_in_progress' && t.advance().error === 'visitor_to_act' &&
      t.act('bogus').error === 'unknown_action' && t.act(illegal).error === 'illegal_action';
  t.delete();
  if (Mod.decideObservation) // C++ exceptions thrown and caught inside the module
    for (let i = 0; i < N; i++) counts.parse += !Mod.decideObservation(i % 10 ? '[' : '['.repeat(900), false).ok;
  return counts;
}
{
  const sp = T.stackPointer();
  const t = refuseMany(T);
  const after = T.stackPointer();
  const m = refuseMany(M);
  for (const [name, c] of [['test', t], ['release', m]])
    for (const [kind, n] of Object.entries(c))
      if (n !== 100000 && !(name === 'release' && kind === 'parse')) fail(`${name} module: ${n} of 100,000 ${kind} refusals as expected`);
  if (after !== sp) fail(`the test module's stack pointer moved from ${sp} to ${after} over its refusals`);
  console.log(`refusals: 100,000 each of refused constructions, calls before a hand, calls during a hand (startHand, advance, ` +
    `unknown and illegal act) on both modules, and refused observations on the test module; stack pointer ${sp} before, ${after} after`);
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

// Every card-like text ([2-9TJQKA][hdcs]) inside any string of a view or a move, wherever it is
// (a card hidden in the log's street name is found). Keys are not scanned ("toAct" holds "Ac"), so
// every key must be one of the view's and the moves' own.
const CARD = /[2-9TJQKA][hdcs]/g;
const KEYS = new Set(['hand', 'over', 'aborted', 'street', 'visitor', 'button', 'smallBlind', 'bigBlind', 'pot', 'toAct',
  'board', 'cards', 'seats', 'seat', 'stack', 'streetContribution', 'totalContribution', 'status', 'legal', 'action',
  'chips', 'to', 'log', 'result', 'pots', 'amount', 'eligible', 'winners', 'shares', 'net', 'showdown', 'ok', 'error', 'message']);
const cardsIn = (value, out = new Set()) => {
  if (typeof value === 'string') for (const c of value.match(CARD) ?? []) out.add(c);
  else if (value && typeof value === 'object')
    for (const [k, v] of Object.entries(value)) {
      if (!Array.isArray(value) && !KEYS.has(k)) out.add(`(unknown key ${k})`);
      cardsIn(v, out);
    }
  return out;
};

function replay() {
  const table = new M.Table(SEED);
  const next = mulberry32(VISITOR_SEED);
  const out = [];
  let leaks = 0;
  for (let h = 0; h < HANDS; h++) {
    if (!table.startHand().ok) throw new Error('startHand refused after a finished hand');
    let v = table.view();
    out.push(`hand ${v.hand} button ${v.button} cards ${v.cards.join(' ')}`);
    const mine = v.cards.join();
    let board = '';
    for (let m = null; ; ) {
      // the only cards a view or a move may hold: the board, the visitor's own two (the same all
      // hand) and, at the end, the showdown seats'; the board only grows
      const allowed = new Set([...v.board, ...v.cards, ...(v.result?.showdown ?? []).flatMap((s) => s.cards)]);
      for (const c of [...cardsIn(v), ...cardsIn(m)]) if (!allowed.has(c)) leaks++;
      for (const s of v.result?.showdown ?? []) if (v.seats[s.seat].status === 'folded') leaks++;
      if (v.cards.length !== 2 || v.cards.join() !== mine || !v.board.join().startsWith(board) || v.board.length > 5) leaks++;
      board = v.board.join();
      if (v.over) break;
      m = v.toAct === v.visitor ? table.act(v.legal[next() % v.legal.length].action) : table.advance();
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
    expect('startHand after a hand', table.startHand().ok, true);
    let v = table.view();
    while (!v.over) {
      expect('startHand during a hand', table.startHand(), { ok: false, error: 'hand_in_progress', over: false, toAct: v.toAct });
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
  for (const [label, make] of [['new Table()', () => new M.Table()], ['Table(1) without new', () => M.Table(1)]]) {
    try {
      make().delete();
      fail(`${label} did not throw`);
    } catch (e) {
      if (!(e instanceof TypeError)) fail(`${label} threw ${e?.name}, expected a TypeError`);
    }
  }
  if ('_networkError' in M || '_constructionFailed' in M) fail('web/post.js did not run: its hooks are still on the module');
  // documented: extra constructor arguments are ignored; a subclass gets embind's objects; every
  // method of a deleted Table throws embind's BindingError
  {
    const deal = (t) => (t.startHand(), t.view().cards.join());
    const [a, b] = [new M.Table(1, 2), new M.Table(1)];
    if (deal(a) !== deal(b)) fail('new Table(1, 2) does not deal as new Table(1)');
    class X extends M.Table {}
    const x = new X(3);
    if (x instanceof X || !(x instanceof M.Table)) fail('a subclass of Table: instances are not as documented');
    for (const t of [a, b, x]) t.delete();
    for (const [name, call] of [['view', () => a.view()], ['startHand', () => a.startHand()], ['act', () => a.act('fold')],
                                ['advance', () => a.advance()], ['delete', () => a.delete()]]) {
      try {
        call();
        fail(`${name}() on a deleted Table did not throw`);
      } catch (e) {
        if (e?.name !== 'BindingError') fail(`${name}() on a deleted Table threw ${e?.name}, expected BindingError`);
      }
    }
  }
  const o = JSON.parse(readFileSync(`${ROOT}weights/demo/${RUN}.parity.jsonl`, 'utf8').split('\n')[0]).observation;
  const obs = JSON.stringify(o);
  for (const [label, got] of [
    ['decideObservation(object)', T.decideObservation(o, false)],
    ['decideObservation(json, "true")', T.decideObservation(obs, 'true')],
    ['decideObservation(json, 1)', T.decideObservation(obs, 1)],
    ['decideObservation("{}")', T.decideObservation('{}', false)],
    ['decideObservation("[")', T.decideObservation('[', false)],
    ['decideObservation(viewer out of range)', T.decideObservation(JSON.stringify({ ...o, viewer_id: 9 }), false)],
    ['decideObservation(bad card)', T.decideObservation(JSON.stringify({ ...o, hole: [[99, 1], [8, 4]] }), false)],
    ['decideObservation(5 seats)', T.decideObservation(JSON.stringify({ ...o, stack: [...o.stack, 200] }), false)],
    ['decideObservation(900 nested arrays)', T.decideObservation('['.repeat(900), false)],
    ['decideObservation(100,000 nested arrays)', T.decideObservation('['.repeat(100000), false)],
  ])
    if (got.ok !== false || typeof got.error !== 'string') fail(`${label}: ${JSON.stringify(got)}, expected a refusal`);
  // Table's test seam: an unchecked illegal action makes the engine throw inside play(); the
  // result has the documented shape, the hand is aborted, and the table plays on
  {
    const t = new T.Table(19);
    const names = T.Table.format().actions;
    let forced = 0;
    for (let h = 0; h < 20; h++) {
      t.startHand();
      for (let v = t.view(); !v.over; v = t.view()) {
        const bad = v.toAct === v.visitor ? names.find((a) => !v.legal.some((l) => l.action === a)) : undefined;
        if (bad === undefined) {
          v.toAct === v.visitor ? t.act(v.legal[0].action) : t.advance();
          continue;
        }
        const r = T.playUnchecked(t, names.indexOf(bad));
        forced++;
        if (Object.keys(r).sort().join() !== 'error,message,ok,over,toAct' || r.ok !== false || r.error !== 'internal_error' ||
            typeof r.message !== 'string' || !r.message || r.over !== true || r.toAct !== -1)
          fail(`forced internal error: ${JSON.stringify(r)}`);
        const after = t.view();
        if (!after.over || !after.aborted || after.result !== null) fail('a forced internal error did not abort the hand');
        break;
      }
    }
    t.startHand();
    for (let v = t.view(); !v.over; v = t.view()) v.toAct === v.visitor ? t.act(v.legal[0].action) : t.advance();
    if (!forced || !t.view().result) fail(`forced internal errors: ${forced}, then no hand played to a result`);
    console.log(`internal errors: ${forced} forced through the test seam, each { ok: false, error: 'internal_error', message, over: true, toAct: -1 }; the table played on`);
    t.delete();
  }
  // the test module still works after those refusals
  const after = new T.Table(3);
  after.startHand();
  while (!after.view().over) after.view().toAct === 0 ? after.act(after.view().legal[0].action) : after.advance();
  if (!after.view().result) fail('the test module could not play a hand after the refused observations');
  after.delete();
  console.log(`binding: ${outOfTurn} out-of-turn and ${illegal} illegal actions refused with the view unchanged; wrong argument types refused`);
}

// 5. out of memory, on a fresh instance of the release module: Tables are made until the 2 GiB
// memory is full. The constructor must then throw post.js's RangeError (it used to return an object
// whose every method threw BindingError), and once the Tables are deleted a new one plays a hand.
{
  const R = await createModule();
  const live = [];
  let error;
  try {
    while (live.length < 2e6) live.push(new R.Table(live.length)); // 2 GiB holds about 705,000
  } catch (e) {
    error = e;
  }
  if (!(error instanceof RangeError) || !/out of memory/.test(error.message))
    fail(`out of memory after ${live.length} Tables: ${error?.name} ${error?.message}`);
  for (const t of live) {
    try {
      t.delete();
    } catch {} // the null-pointer objects of the old behaviour
  }
  const t = new R.Table(1);
  t.startHand();
  for (let v = t.view(); !v.over; v = t.view()) v.toAct === v.visitor ? t.act(v.legal[0].action) : t.advance();
  if (!t.view().result) fail('after the Tables were deleted, a new one did not play a hand');
  t.delete();
  console.log(`out of memory: ${live.length} live Tables, then new Table threw ${error?.name} "${error?.message}"; after deleting them a new Table played a hand`);
}

if (failures) {
  console.error(`FAIL: ${failures} failure(s)`);
  process.exit(1);
}
console.log('PASS: parity records, replay identical to the native build, totals as pinned, hidden cards and refusals as expected');
