"""Minimal stand-in for dm-tree's map_structure (only what tapnet.torch uses)."""


def map_structure(fn, *structs):
    first = structs[0]
    if isinstance(first, dict):
        return {k: map_structure(fn, *[s[k] for s in structs]) for k in first}
    if isinstance(first, tuple) and hasattr(first, '_fields'):
        return type(first)(*[map_structure(fn, *parts) for parts in zip(*structs)])
    if isinstance(first, (list, tuple)):
        return type(first)(map_structure(fn, *parts) for parts in zip(*structs))
    return fn(*structs)
