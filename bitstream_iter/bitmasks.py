"""Bit masks for integers.

Bit masks provide a flexible way of converting between bits and integers.
They support arbitrary integer widths, signed values, and any bit ordering.
Some use cases may filter, duplicate, or merge bits using special masks.

A few standard masks are provided for convenience:

 INT8_MSB_FIRST - 8-bit signed, most significant bit first.
 INT8_LSB_FIRST - 8-bit signed, least significant bit first.
UINT8_MSB_FIRST - 8-bit unsigned, most significant bit first.
UINT8_LSB_FIRST - 8-bit unsigned, least significant bit first.

 INT16_BE - 16-bit signed, big-endian byte order
 INT16_LE - 16-bit signed, little-endian byte order
UINT16_BE - 16-bit unsigned, big-endian byte order
UINT16_LE - 16-bit unsigned, little-endian byte order

 INT32_BE - 32-bit signed, big-endian byte order
 INT32_LE - 32-bit signed, little-endian byte order
UINT32_BE - 32-bit unsigned, big-endian byte order
UINT32_LE - 32-bit unsigned, little-endian byte order

 INT64_BE - 64-bit signed, big-endian byte order
 INT64_LE - 64-bit signed, little-endian byte order
UINT64_BE - 64-bit unsigned, big-endian byte order
UINT64_LE - 64-bit unsigned, little-endian byte order

The 16, 32, and 64-bit masks use most significant bit first in each byte.
Custom masks can be created manually, or using create().
"""

import itertools
from typing import Any, Iterable, Iterator, Sequence, Tuple


def create(width: int, signed: bool, lsb_first: bool = False,
           reverse_group_width: int = None) -> Tuple[int, ...]:
    """Create bit masks for any integer type.

    Args:
        width: Number of bits in the integer.

        signed: Whether the integer is signed. If so, the most significant bit
            mask will be negative to provide sign extension when combining.

        lsb_first: Whether the least significant bit is first.

        reverse_group_width: As the last step, group widths of bits and reverse
            the order of all groups. Can be used for byte order swapping when
            correctly matched with msb_first.

    Returns:
        Integer bit masks.
    """

    bits = list((1 << n) for n in range(width))

    if signed:
        bits.append(-bits.pop())

    if not lsb_first:
        bits.reverse()

    if not reverse_group_width:
        return tuple(bits)

    group_starts = reversed(range(0, width, reverse_group_width))
    groups = (bits[s:s + reverse_group_width] for s in group_starts)
    return tuple(itertools.chain.from_iterable(groups))


def apply(masks: Sequence[int], value: int) -> Iterator[int]:
    """Apply each bit mask to the given value using bitwise AND.

    Args:
        masks: Bit masks to apply.

        value: Value to mask.

    Returns:
        Iterator with each mask independently applied to the value.
    """
    return ((value & mask) for mask in masks)


def combine(masks: Sequence[int], bits: Iterable[Any]) -> int:
    """Combine bit masks for selected bits using bitwise OR.

    Args:
        masks: Bit masks to selectively combine.

        bits: Bits that select which bit masks to combine.

    Returns:
        Integer value formed by combining the selected bit masks.
    """
    value = 0

    for bit, mask in zip(bits, masks):
        if bit:
            value |= mask

    return value


INT8_MSB_FIRST = create(width=8, signed=True)
INT8_LSB_FIRST = create(width=8, signed=True, lsb_first=True)
UINT8_MSB_FIRST = create(width=8, signed=False)
UINT8_LSB_FIRST = create(width=8, signed=False, lsb_first=True)

INT16_BE = create(width=16, signed=True)
INT16_LE = create(width=16, signed=True, reverse_group_width=8)
UINT16_BE = create(width=16, signed=False)
UINT16_LE = create(width=16, signed=False, reverse_group_width=8)

INT32_BE = create(width=32, signed=True)
INT32_LE = create(width=32, signed=True, reverse_group_width=8)
UINT32_BE = create(width=32, signed=False)
UINT32_LE = create(width=32, signed=False, reverse_group_width=8)

INT64_BE = create(width=64, signed=True)
INT64_LE = create(width=64, signed=True, reverse_group_width=8)
UINT64_BE = create(width=64, signed=False)
UINT64_LE = create(width=64, signed=False, reverse_group_width=8)
