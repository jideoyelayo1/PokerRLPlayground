import poker_engine as pe
import inspect

def banner(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

def is_pybind_enum_value(obj):
    # pybind enum values usually have `.name` and `.value`
    return hasattr(obj, "name") and hasattr(obj, "value")

def inspect_pybind_enum(enum_cls):
    print(f"\nEnum: {enum_cls.__name__}")
    names = []
    for n in dir(enum_cls):
        if n.startswith("_"):
            continue
        v = getattr(enum_cls, n)
        if is_pybind_enum_value(v):
            names.append(n)

    if not names:
        print("  (No enum values found via dir())")
        return

    for n in sorted(names):
        v = getattr(enum_cls, n)
        print(f"  {n} -> name={getattr(v, 'name', None)} value={getattr(v, 'value', None)} repr={v}")

def inspect_class(cls):
    print(f"\nClass: {cls.__name__}")
    print("-" * 60)

    try:
        print("Constructor:", inspect.signature(cls))
    except Exception:
        print("Constructor: <unavailable>")

    for name, member in inspect.getmembers(cls):
        if name.startswith("__"):
            continue
        if inspect.isfunction(member) or inspect.ismethod(member):
            try:
                print(f"  method: {name}{inspect.signature(member)}")
            except Exception:
                print(f"  method: {name}(...)")
        else:
            # avoid dumping huge internal descriptors
            if name in ("__dict__", "__weakref__"):
                continue
            print(f"  attr:   {name}")

banner("MODULE INFO")
print(pe)
print("Module name:", pe.__name__)
print("Module file:", pe.__file__)

banner("DIR(poker_engine)")
for name in dir(pe):
    print(name)

banner("ENUMS (pybind-safe)")
inspect_pybind_enum(pe.ActionType)
inspect_pybind_enum(pe.RaiseSize)
inspect_pybind_enum(pe.Suits)
inspect_pybind_enum(pe.Ranks)
inspect_pybind_enum(pe.HandCat)

banner("CLASSES")
for cls in [pe.Card, pe.Action, pe.ActionSpec, pe.Pot, pe.HandRank, pe.Player, pe.GameEngine, pe.Observation]:
    inspect_class(cls)

banner("FREE FUNCTIONS")
for name, obj in inspect.getmembers(pe):
    if inspect.isbuiltin(obj) or inspect.isfunction(obj):
        if name.startswith("_"):
            continue
        try:
            print(f"{name}{inspect.signature(obj)}")
        except Exception:
            print(f"{name}(...)")

banner("SANITY CHECK")
print("pe.ActionType.Raise =", pe.ActionType.Raise)
print("pe.RaiseSize.Pot   =", pe.RaiseSize.Pot)

# Show that the values are accessible even if enum isn't iterable
print("Raise name/value:", pe.ActionType.Raise.name, pe.ActionType.Raise.value)
print("Pot name/value:  ", pe.RaiseSize.Pot.name, pe.RaiseSize.Pot.value)

banner("DONE")
