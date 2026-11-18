// Linked into the module with --post-js (web/build.sh). With Emscripten 6.0.13 it runs inside the
// module factory after the runtime is up, just before the factory returns. It replaces Module.Table
// with a constructor that checks the seed here, in JavaScript, before anything enters WebAssembly,
// and throws the TypeError from here (web/test.mjs fails if this did not happen).
//
// Nothing may be thrown from C++ into JavaScript: such an exception unwinds past WebAssembly without
// restoring its stack pointer, so each one leaks stack until the module fails (4,499 refused seeds
// did it when the C++ side threw). The bound calls return refusals instead (web/poker_wasm.cpp).
{
  const Raw = Module['Table'];
  const networkError = Module['_networkError']();
  const constructionFailed = Module['_constructionFailed'];
  delete Module['_networkError'];
  delete Module['_constructionFailed'];
  function Table(seed) {
    if (!new.target) throw new TypeError("Table: use 'new Table(seed)'");
    if (networkError) throw new Error(`Table: the network did not load: ${networkError}`);
    if (typeof seed !== 'number' || !Number.isSafeInteger(seed) || seed < 0)
      throw new TypeError('Table: the seed must be a whole number from 0 to 2^53 - 1');
    // cleared first: a failed construction through the embind class (below), which this check
    // bypasses, leaves it set, and the next Table would be dropped with a spurious RangeError
    constructionFailed();
    const table = new Raw(seed);
    // out of memory: the C++ constructor returned a null pointer, and every method of that object
    // would throw embind's BindingError (only that sets the flag, cleared just above: a Table that was
    // built is never dropped here)
    if (constructionFailed())
      throw new RangeError('Table: out of memory; delete() Tables that are no longer used and try again');
    return table;
  }
  // The embind class stays reachable as Object.getPrototypeOf(table).constructor. It does not throw
  // either: at -O2 embind's JavaScript converts any argument to a number, as Number() would ('5'
  // plays seed 5, true seed 1, [3] seed 3), and the C++ side truncates a fraction and turns anything
  // outside 0 to 2^53 - 1 ({}, null, undefined, NaN, negatives, infinities, 2^60) into seed 0.
  Table.prototype = Raw.prototype;
  Table.format = Raw.format;
  Module['Table'] = Table;
}
