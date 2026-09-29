"""Static cost scan of a checkpoint pickle, run before pickle.loads (R6-M01).

Unpickling builds Python objects whose size has no fixed ratio to their
pickled bytes: an int costs 2-5 pickle bytes and ~40 traced bytes, an empty
set 2 pickle bytes and ~235 traced bytes (measured on CPython 3.11 (tracemalloc)). A factor on
the payload length is therefore not a bound. This module walks the opcode
stream without building any object, and returns a worst-case charge for what
pickle.loads will allocate.

The walk also refuses everything a checkpoint does not need. A bound is only
meaningful if no pickled callable can allocate on its own: a 30-byte stream
that REDUCEs builtins.bytearray(10**9) allocates 1 GB. So the only callables
admitted are numpy.dtype (plain bool, int, uint, float and complex codes only,
R8-M01) and numpy's _frombuffer, with argument shapes that
cannot allocate more than their inputs, and BUILD is admitted only on a dtype.

The per-object charges cover the peaks measured on CPython 3.11 (tracemalloc)
for lists, dicts, sets, tuples, strs, bytes, ints and floats, including
dict/set resize and the unpickler's memo and stack growth. A str is charged
per UTF-8 byte for its worst decode, not its average (R7-M01, see STR_BYTES_PER_UTF8_BYTE).

The charges also cover the scan's own bookkeeping (R7-M02). Stateless
values share one tag per kind, so a list, dict, set, empty tuple, scalar or
long str costs the scan one 8-byte stack slot; only short strs, small tuples,
globals and buffers get their own tag (under 500 bytes, always under their
charge). Every push onto the stack, the mark stack or the memo is checked
against the limit before it happens, so a scan stopped at a limit has
allocated less than the limit. tests/test_r6_m01_metadata_memory_bound.py and
tests/test_r7_m01_m02_unicode_and_scanner_bound.py measure real loads and
scans against them.

Measurement environment: every per-object constant below (SCALAR_BYTES,
VARSIZE_HEADER_BYTES, LIST_BYTES, SET_BYTES, TUPLE_BYTES, HASH_SLOT_BYTES,
etc.) is a peak measured on CPython 3.11 (tracemalloc) (see the test files
above for the exact measurement harness). They cover CPython's own object
headers, PEP 393 string storage, and container over-allocation as that
build lays them out; a margin is added where a decoder or resize path can
transiently hold more than the object's final size (STR_BYTES_PER_UTF8_BYTE
above). They are not re-verified against other CPython minor versions or
against alternative implementations (PyPy, GraalPy): object header size and
container growth factors can change release to release, so a version that
allocates more per object than measured here would need these constants
re-measured and raised, not merely re-labeled. The only measured baseline
is CPython 3.11 (tracemalloc); no other version is claimed.
"""

import re
import struct
from array import array
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

# Hard cap on objects one checkpoint may create. It bounds the scan's own
# CPU time; the byte charge below is what bounds memory.
MAX_UNPICKLE_OBJECTS = 1 << 20

SCALAR_BYTES = 64          # int, float: 24-32 traced bytes each
VARSIZE_HEADER_BYTES = 160  # str / bytes / bytearray header, data charged on top
# PEP 393 stores a str at 1, 2 or 4 bytes per char, set by its widest char,
# so one char past U+FFFF makes an ASCII str 4 bytes per UTF-8 byte. The
# decoder is worse (R7-M01): it widens its buffer in place, holding the
# 2-byte and the new 4-byte copy at once (6n), and a surrogate that
# "surrogatepass" handles builds a UnicodeDecodeError holding a copy of the
# whole input (n). "a"*N + "\ud800" + "\U00010000" peaks at 7.001n on
# CPython 3.11.16; 8n keeps one n of margin.
STR_BYTES_PER_UTF8_BYTE = 8
LIST_BYTES = 128           # empty list or dict object with GC header
SET_BYTES = 384            # empty set or frozenset (216 traced bytes)
TUPLE_BYTES = 160          # plus TUPLE_SLOT_BYTES per item
TUPLE_SLOT_BYTES = 16
LIST_SLOT_BYTES = 32       # over-allocation plus APPENDS' temporary slice
HASH_SLOT_BYTES = 192      # dict / set entry, old and new table alive on resize
MEMO_SLOT_BYTES = 32       # memo array doubles on growth
STACK_SLOT_BYTES = 32      # unpickler value and mark stacks
CALL_BYTES = 2048          # one dtype or _frombuffer call, or a global lookup
UNPICKLER_BYTES = 4096     # the unpickler itself, its initial memo and stacks (~900 traced)

# Globals a checkpoint may name. numpy 1.x pickles _frombuffer under
# numpy.core, numpy 2.x under numpy._core.
_DTYPE = "dtype"
_FROMBUFFER = "frombuffer"
_DTYPE_FRESH, _DTYPE_SEALED = "fresh", "sealed"  # dtype tag states, see BUILD
_ALLOWED_GLOBALS = {
    ("numpy", "dtype"): _DTYPE,
    ("numpy.core.numeric", "_frombuffer"): _FROMBUFFER,
    ("numpy._core.numeric", "_frombuffer"): _FROMBUFFER,
}
_MAX_NAME_LEN = 64       # longest str kept for global names and dtype codes
_MAX_DTYPE_CODE_LEN = 32  # "f4", "<U16", "M8[ns]"; a long code can define many fields
_MAX_SMALL_TUPLE = 32    # longest tuple whose items are tracked (shapes, call args)
# dtype codes a checkpoint may name: the codes numpy's own dtype.__reduce__
# writes for bool, int, uint, float and complex. Byte order travels in the
# BUILD state, not the code. Anything else is refused (R8-M01): "(1024,)u1"
# is a subarray dtype, so frombuffer(...).reshape(shape, "F") copies the
# whole buffer, and "i4,f4" is a structured dtype.
_SCALAR_DTYPE_CODE = re.compile(r"(?:b1|[iu][1248]|f[248]|c(?:8|16))\Z")


class PickleRefusedError(ValueError):
    """The pickle uses an opcode, global or call shape a checkpoint never
    needs, so its allocation cannot be bounded. Nothing was unpickled."""
    pass


class MemoryBoundExceededError(MemoryError):
    """The worst-case allocation of unpickling exceeds what the checkpoint
    header reserved for it. Raised before pickle.loads runs."""

    def __init__(self, message: str, cost: "UnpickleCost") -> None:
        super().__init__(message)
        self.cost = cost


@dataclass(frozen=True)
class UnpickleCost:
    """Worst-case allocation of one pickle.loads call.

    array_bytes: data of the buffers that numpy arrays wrap without copying;
        they become the loaded core's weights, so the resident size covers them.
    object_bytes: every other object the unpickler builds (metadata, the
        array and dtype objects, unwrapped buffers, memo and stack).
    objects: number of objects created.
    """
    array_bytes: int
    object_bytes: int
    objects: int


class _Tag:
    """Abstract value on the simulated stack: what kind of object the real
    unpickler would hold there, never the object itself."""
    __slots__ = ("kind", "value", "items", "nbytes", "wrapped")

    def __init__(self, kind: str, value: Any = None, items: Optional[Sequence["_Tag"]] = None, nbytes: int = 0):
        self.kind = kind
        self.value = value
        self.items = items
        self.nbytes = nbytes
        self.wrapped = False


_SCALAR_KINDS = ("int", "float", "bool", "none")
# Values without per-object state share one tag per kind, so the scan's
# bookkeeping for them is one stack pointer (R7-M02: a fresh tag per EMPTY_TUPLE
# cost the scan 152 bytes each, 15 MB for tuple([()] * 100000)). A shared
# tag is never mutated: the only mutations, `wrapped` on buffer tags and
# `value` on dtype tags, are on per-object tags.
_SHARED_TAGS = {kind: _Tag(kind) for kind in _SCALAR_KINDS + (
    "list", "dict", "set", "frozenset", "ndarray", "str")}
_EMPTY_TUPLE_TAG = _Tag("tuple", items=())
_LARGE_TUPLE_TAG = _Tag("tuple")  # items not tracked
_BUFFER_KINDS = ("bytes", "bytearray")


def _is_plain_scalar_dtype(code: str) -> bool:
    """numpy's own verdict on a code the allowlist admitted: no subarray,
    no fields, no object references, and a nonzero item size."""
    import numpy as np
    try:
        dt = np.dtype(code)
    except (TypeError, ValueError):
        return False
    return dt.subdtype is None and dt.names is None and not dt.hasobject and dt.itemsize > 0


def scan_pickle(data: Any, limit: Optional[int] = None) -> UnpickleCost:
    """Walks a protocol 2-5 pickle and returns its worst-case unpickle cost.

    Reads opcode arguments by length and never copies them. The scan's own
    bookkeeping is smaller than the charge it has counted so far, so with a
    limit the scan stops before it allocates more than the limit.

    Args:
        limit: most object_bytes (plus stack) the caller reserved. The scan
            raises as soon as the running charge passes it.

    Raises:
        PickleRefusedError: an opcode, global or call outside the checkpoint
            subset, a malformed stream, or more than MAX_UNPICKLE_OBJECTS objects.
        MemoryBoundExceededError: the running charge passed limit.
    """
    mv = memoryview(data).cast("B")
    end = len(mv)
    pos = 0
    stack: list = []
    # Plain ints would cost the scan an int object per MARK once the stack
    # passes 256 entries; an array holds 8 bytes per mark.
    marks = array("q")
    memo: list = []
    buffers: list = []
    object_bytes = 0
    objects = 0
    max_depth = 0

    def refuse(msg: str) -> PickleRefusedError:
        return PickleRefusedError(f"pickle offset {pos}: {msg}")

    def take(n: int) -> memoryview:
        nonlocal pos
        if n < 0 or pos + n > end:
            raise refuse("truncated argument")
        out = mv[pos:pos + n]
        pos += n
        return out

    def uint(n: int) -> int:
        return int.from_bytes(take(n), "little")

    def pop() -> _Tag:
        if not stack or (marks and marks[-1] == len(stack)):
            raise refuse("stack underflow")
        return stack.pop()

    def pop_mark() -> Tuple[int, Optional[list]]:
        """Pops to the last mark. Returns the item count, and the items only
        when a small tuple needs them: a slice of every item would cost the
        scan another 8 bytes per stack slot."""
        if not marks:
            raise refuse("no mark on the stack")
        k = marks.pop()
        n = len(stack) - k
        items = stack[k:] if n <= _MAX_SMALL_TUPLE else None
        # del stack[k:] with k > 0 mallocs a pointer array the size of the
        # deleted run; chunks cap it at 32 KB.
        while len(stack) > k:
            del stack[max(k, len(stack) - 4096):]
        return n, items

    def top(kind: str, op: str) -> _Tag:
        if not stack or (marks and marks[-1] == len(stack)) or stack[-1].kind != kind:
            raise refuse(f"{op} applies only to a {kind} here")
        return stack[-1]

    def over_limit(total: int) -> MemoryBoundExceededError:
        return MemoryBoundExceededError(
            f"pickle offset {pos}: unpickling needs more than the {limit} bytes reserved "
            f"for its objects ({objects} objects so far); nothing was unpickled.",
            UnpickleCost(array_bytes=0, object_bytes=total, objects=objects),
        )

    def charge(cost: int) -> None:
        nonlocal object_bytes
        object_bytes += cost
        if limit is not None and object_bytes + STACK_SLOT_BYTES * max_depth > limit:
            raise over_limit(object_bytes)

    def grow() -> None:
        """Checks one more stack or mark slot before it is appended: the
        charge covers the unpickler's slot and the scan's own."""
        nonlocal max_depth
        depth = len(stack) + len(marks) + 1
        if depth > max_depth:
            if depth > MAX_UNPICKLE_OBJECTS:
                raise refuse(f"stack deeper than {MAX_UNPICKLE_OBJECTS}")
            max_depth = depth
            charge(0)

    def push(tag: _Tag) -> None:
        grow()
        stack.append(tag)

    def create(tag: _Tag, cost: int) -> None:
        nonlocal objects
        objects += 1
        if objects > MAX_UNPICKLE_OBJECTS:
            raise refuse(f"more than {MAX_UNPICKLE_OBJECTS} objects")
        charge(cost)
        push(tag)

    def varsize(kind: str, n: int, keep_text: bool) -> None:
        raw = take(n)
        if kind in _BUFFER_KINDS:
            tag = _Tag(kind, nbytes=n)
            create(tag, VARSIZE_HEADER_BYTES)
            buffers.append(tag)  # one slot per object, under the header charge
            return
        # Charge first: a short str's tag and decoded value are the scan's
        # only per-str allocations, and they fit in the charge.
        charge(VARSIZE_HEADER_BYTES + STR_BYTES_PER_UTF8_BYTE * n)
        tag = _SHARED_TAGS["str"]
        if keep_text and n <= _MAX_NAME_LEN:
            try:
                tag = _Tag("str", value=bytes(raw).decode("utf-8", "surrogatepass"))
            except UnicodeDecodeError:
                raise refuse("invalid UTF-8 in a str")
        create(tag, 0)

    def small_tuple(items: Optional[list]) -> _Tag:
        return _LARGE_TUPLE_TAG if items is None else _Tag("tuple", items=items)

    def short_str(tag: _Tag, limit: int) -> bool:
        return tag.kind == "str" and tag.value is not None and len(tag.value) <= limit

    def reduce(func: _Tag, args: _Tag) -> _Tag:
        if func.kind != "global":
            raise refuse(f"REDUCE on a {func.kind}, not an allowed global")
        items = args.items if args.kind == "tuple" else None
        if items is None:
            raise refuse("REDUCE arguments are not a small tuple")
        if func.value == _DTYPE:
            # numpy.dtype(code, align, copy): only a short type code, never a
            # field list, so the descriptor stays a few hundred bytes.
            if not (1 <= len(items) <= 3 and short_str(items[0], _MAX_DTYPE_CODE_LEN)
                    and all(t.kind in _SCALAR_KINDS for t in items[1:])):
                raise refuse("numpy.dtype takes only a short type code here")
            if not (_SCALAR_DTYPE_CODE.match(items[0].value) and _is_plain_scalar_dtype(items[0].value)):
                raise refuse(f"subarray or compound dtype not permitted in _frombuffer: {items[0].value!r}")
            # Per-object, under CALL_BYTES: BUILD may set its state once,
            # before any array uses it (see BUILD below).
            return _Tag("dtype", value=_DTYPE_FRESH)
        # _frombuffer(buf, dtype, shape, order) is frombuffer(...).reshape().
        # With a plain scalar dtype, frombuffer gives a 1-D contiguous array,
        # and reshaping that in C or F order is always a view: the array owns
        # no data beyond buf. A subarray dtype would make it 2-D and an F
        # reshape a full copy (R8-M01), so reduce() refuses those above.
        if not (len(items) == 4 and items[0].kind in _BUFFER_KINDS and items[1].kind == "dtype"
                and items[2].kind == "tuple" and items[2].items is not None
                and all(t.kind == "int" for t in items[2].items)
                and short_str(items[3], 1) and items[3].value in ("C", "F")):
            raise refuse("_frombuffer takes (buffer, dtype, int shape, 'C' or 'F') only")
        items[0].wrapped = True
        items[1].value = _DTYPE_SEALED
        return _SHARED_TAGS["ndarray"]

    op = b""
    while True:
        if pos >= end:
            raise refuse("stream ends without STOP")
        op = mv[pos:pos + 1].tobytes()
        pos += 1

        if op == b"\x80":  # PROTO
            proto = uint(1)
            if not 2 <= proto <= 5:
                raise refuse(f"protocol {proto} is not supported")
        elif op == b"\x95":  # FRAME: the in-memory unpickler reads frames in place
            uint(8)
        elif op == b".":  # STOP
            if len(stack) != 1 or marks:
                raise refuse("STOP with a malformed stack")
            if pos != end:
                raise refuse(f"{end - pos} bytes after STOP")
            break
        elif op == b"(":  # MARK
            grow()
            marks.append(len(stack))
        elif op == b"0":  # POP: pops the mark itself when nothing sits above it
            if marks and marks[-1] == len(stack):
                marks.pop()
            else:
                pop()
        elif op == b"1":  # POP_MARK
            pop_mark()
        elif op == b"2":  # DUP
            tag = pop()
            stack.append(tag)
            push(tag)
        elif op == b"\x94":  # MEMOIZE
            if not stack:
                raise refuse("MEMOIZE on an empty stack")
            if len(memo) >= MAX_UNPICKLE_OBJECTS:
                raise refuse(f"more than {MAX_UNPICKLE_OBJECTS} memo entries")
            charge(MEMO_SLOT_BYTES)
            memo.append(stack[-1])
        elif op in (b"h", b"j"):  # BINGET, LONG_BINGET
            idx = uint(1 if op == b"h" else 4)
            if idx >= len(memo):
                raise refuse(f"memo index {idx} was never stored")
            push(memo[idx])
        elif op == b"N":
            create(_SHARED_TAGS["none"], 0)
        elif op in (b"\x88", b"\x89"):  # NEWTRUE, NEWFALSE
            create(_SHARED_TAGS["bool"], 0)
        elif op in (b"K", b"M", b"J"):  # BININT1, BININT2, BININT
            take({b"K": 1, b"M": 2, b"J": 4}[op])
            create(_SHARED_TAGS["int"], SCALAR_BYTES)
        elif op in (b"\x8a", b"\x8b"):  # LONG1, LONG4
            n = uint(1) if op == b"\x8a" else struct.unpack("<i", take(4))[0]
            take(n)
            create(_SHARED_TAGS["int"], SCALAR_BYTES + 2 * n)
        elif op == b"G":  # BINFLOAT
            take(8)
            create(_SHARED_TAGS["float"], SCALAR_BYTES)
        elif op == b"\x8c":  # SHORT_BINUNICODE
            varsize("str", uint(1), True)
        elif op == b"X":  # BINUNICODE
            varsize("str", uint(4), True)
        elif op == b"\x8d":  # BINUNICODE8
            varsize("str", uint(8), True)
        elif op == b"C":  # SHORT_BINBYTES
            varsize("bytes", uint(1), False)
        elif op == b"B":  # BINBYTES
            varsize("bytes", uint(4), False)
        elif op == b"\x8e":  # BINBYTES8
            varsize("bytes", uint(8), False)
        elif op == b"\x96":  # BYTEARRAY8
            varsize("bytearray", uint(8), False)
        elif op == b"]":  # EMPTY_LIST
            create(_SHARED_TAGS["list"], LIST_BYTES)
        elif op == b"}":  # EMPTY_DICT
            create(_SHARED_TAGS["dict"], LIST_BYTES)
        elif op == b"\x8f":  # EMPTY_SET
            create(_SHARED_TAGS["set"], SET_BYTES)
        elif op == b")":  # EMPTY_TUPLE: a singleton for the unpickler too
            push(_EMPTY_TUPLE_TAG)
        elif op in (b"\x85", b"\x86", b"\x87"):  # TUPLE1..3
            n = op[0] - 0x84
            items = [pop() for _ in range(n)][::-1]
            create(small_tuple(items), TUPLE_BYTES + TUPLE_SLOT_BYTES * n)
        elif op == b"t":  # TUPLE
            n, items = pop_mark()
            create(small_tuple(items), TUPLE_BYTES + TUPLE_SLOT_BYTES * n)
        elif op == b"l":  # LIST
            n, _ = pop_mark()
            create(_SHARED_TAGS["list"], LIST_BYTES + LIST_SLOT_BYTES * n)
        elif op == b"d":  # DICT
            n, _ = pop_mark()
            if n % 2:
                raise refuse("DICT with an odd item count")
            create(_SHARED_TAGS["dict"], LIST_BYTES + HASH_SLOT_BYTES * (n // 2))
        elif op == b"\x91":  # FROZENSET
            n, _ = pop_mark()
            create(_SHARED_TAGS["frozenset"], SET_BYTES + HASH_SLOT_BYTES * n)
        elif op == b"a":  # APPEND
            pop()
            top("list", "APPEND")
            charge(LIST_SLOT_BYTES)
        elif op == b"e":  # APPENDS
            n, _ = pop_mark()
            top("list", "APPENDS")
            charge(LIST_SLOT_BYTES * n)
        elif op == b"s":  # SETITEM
            pop()
            pop()
            top("dict", "SETITEM")
            charge(HASH_SLOT_BYTES)
        elif op == b"u":  # SETITEMS
            n, _ = pop_mark()
            if n % 2:
                raise refuse("SETITEMS with an odd item count")
            top("dict", "SETITEMS")
            charge(HASH_SLOT_BYTES * (n // 2))
        elif op == b"\x90":  # ADDITEMS
            n, _ = pop_mark()
            top("set", "ADDITEMS")
            charge(HASH_SLOT_BYTES * n)
        elif op == b"\x93":  # STACK_GLOBAL
            name = pop()
            module = pop()
            if not (short_str(module, _MAX_NAME_LEN) and short_str(name, _MAX_NAME_LEN)):
                raise refuse("STACK_GLOBAL needs two literal names")
            kind = _ALLOWED_GLOBALS.get((module.value, name.value))
            if kind is None:
                raise refuse(f"global {module.value}.{name.value} is not allowed in a checkpoint")
            create(_Tag("global", value=kind), CALL_BYTES)
        elif op == b"R":  # REDUCE
            args = pop()
            func = pop()
            create(reduce(func, args), CALL_BYTES)
        elif op == b"b":  # BUILD: only dtype.__setstate__ with flat scalar state
            state = pop()
            dtype = top("dtype", "BUILD")
            # One BUILD per dtype, before any array uses it: a second BUILD
            # through the memo could set NPY_ITEM_HASOBJECT on the dtype of
            # an array that already exists over raw bytes.
            if dtype.value != _DTYPE_FRESH:
                raise refuse("BUILD on a dtype that was already built or used")
            dtype.value = _DTYPE_SEALED
            if not (state.kind == "tuple" and state.items is not None and all(
                    t.kind in _SCALAR_KINDS or short_str(t, _MAX_DTYPE_CODE_LEN) for t in state.items)):
                raise refuse("dtype state must be a flat tuple of scalars")
            # (version, byteorder, subdescr, names, fields, ...): a subarray
            # or field list here would turn the scalar dtype compound (R8-M01).
            if not (len(state.items) >= 5 and all(t.kind == "none" for t in state.items[2:5])):
                raise refuse("subarray or compound dtype not permitted in _frombuffer")
            charge(CALL_BYTES)
        else:
            raise refuse(f"opcode {op!r} is not allowed in a checkpoint")

    array_bytes = sum(b.nbytes for b in buffers if b.wrapped)
    object_bytes += UNPICKLER_BYTES
    object_bytes += sum(b.nbytes for b in buffers if not b.wrapped)
    object_bytes += STACK_SLOT_BYTES * max_depth
    # The tail charges were never tested against the limit on the way.
    if limit is not None and object_bytes > limit:
        raise over_limit(object_bytes)
    return UnpickleCost(array_bytes=array_bytes, object_bytes=object_bytes, objects=objects)
