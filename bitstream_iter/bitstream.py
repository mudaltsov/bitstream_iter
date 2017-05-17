import itertools
from typing import Any, Iterable, Iterator, List, NewType, Sequence

from bitstream_iter import bitmasks


Bit = NewType('Bit', int)


def make_bit(value: Any) -> Bit:
    """Convert any value to 0 or 1 using truth value testing."""
    if value:
        return Bit(1)
    return Bit(0)


class BitStream(Iterable[Bit]):
    """Binary stream that uses iterators for I/O at the bit level.

    BitStream() -> empty BitStream object
    BitStream(bits[, byte_bitmasks]) -> BitStream object with bit values
    BitStream.from_bytes(values[, byte_bitmasks]) -> BitStream object with bits
                                                     extracted from byte values
    """

    def __init__(self, bits: Iterable[Any] = None,
                 byte_bitmasks: Sequence[int] = None) -> None:
        """Initialize a new bit stream.

        Args:
            bits: Initial bit values to lazily write to the stream.
                Each value is converted to 0 or 1 by truth value testing.

            byte_bitmasks: Bit masks used for byte <-> bit conversion.
                Default is all 8 bits with most significant bit first.
        """
        self._bit_sources = []  # type: List[Iterator[Bit]]
        self.byte_bitmasks = byte_bitmasks or bitmasks.UINT8_MSB_FIRST

        if bits is not None:
            self.write_bits(bits)

    @classmethod
    def from_bytes(cls, values: Iterable[int],
                   byte_bitmasks: Sequence[int] = None) -> 'BitStream':
        """Create a bit stream containing bytes.

        Args:
            values: Initial byte values to lazily write to the stream.

            byte_bitmasks: Bit masks used for byte <-> bit conversion.
                Default is all 8 bits with most significant bit first.

        Returns:
            A new BitStream initialized with the given data.
        """
        stream = cls(byte_bitmasks=byte_bitmasks)
        stream.write_bytes(values=values)
        return stream

    def __bytes__(self) -> bytes:
        """Implements bytes(self).

        Returns:
            Bytes formed by consuming all possible bits from the stream.
            Bits that don't form a complete byte remain in the stream.
            Uses byte_bitmasks specified in the constructor.
        """
        return bytes(self.iter_bytes())

    def __iter__(self) -> Iterator[Bit]:
        """Implements iter(self).

        Returns:
            Iterator that produces 0 or 1 for each bit consumed from the
            start of the stream, until all bits have been exhausted.
        """
        return itertools.chain.from_iterable(self._bit_sources)

    def iter_bytes(self, masks: Sequence[int] = None) -> Iterator[int]:
        """Iterator that generates bytes from the bit stream.

        Args:
            masks: Bit masks used for combining bits into a byte.
                Default is byte_bitmasks specified in the constructor.

        Returns:
            Iterator that produces each byte by consuming bits from the start
            of the stream, until no more complete bytes are available.
            Bits that don't form a complete byte remain in the stream.
        """
        masks = masks or self.byte_bitmasks
        return self.iter_ints(masks=masks)

    def iter_ints(self, masks: Sequence[int]) -> Iterator[int]:
        """Iterator that generates integers from the bit stream.

        Args:
            masks: Bit masks used for combining bits into an integer.

        Yields:
            Next integer value formed by consuming bits from the start of
            the stream, until no more complete integers are available.
            Bits that don't form a complete integer remain in the stream.
        """
        bit_iter = iter(self)

        while True:
            bits = tuple(itertools.islice(bit_iter, len(masks)))

            if len(bits) < len(masks):
                if len(bits):
                    self.write_bits(bits)
                return

            yield bitmasks.combine(masks=masks, bits=bits)

    def write_bits(self, values: Iterable[Any]) -> 'BitStream':
        """Write lazily accessed bits to the end of the stream.

        Args:
            values: Bit values that are accessed when consuming stream bits.
                Each value is converted to 0 or 1 by truth value testing.

        Returns:
            The same BitStream instance, as a convenience for consuming data.
        """
        self._bit_sources.append(make_bit(b) for b in values)
        return self

    def write_bytes(self, values: Iterable[int],
                    masks: Sequence[int] = None) -> 'BitStream':
        """Write lazily accessed bytes to the end of the stream.

        Args:
            values: Byte values that are accessed when consuming stream bits.

            masks: Bit masks applied to each byte to produce bits.
                Default is byte_bitmasks specified in the constructor.

        Returns:
            The same BitStream instance, as a convenience for consuming data.
        """
        masks = masks or self.byte_bitmasks
        return self.write_ints(values=values, masks=masks)

    def write_ints(self, values: Iterable[int],
                   masks: Sequence[int]) -> 'BitStream':
        """Write lazily accessed integers to the end of the stream.

        Args:
            values: Values that are accessed when consuming stream bits.

            masks: Bit masks applied to each integer to produce bits.

        Returns:
            The same BitStream instance, as a convenience for consuming data.
        """
        masked_values = (bitmasks.apply(masks=masks, value=v) for v in values)
        return self.write_bits(itertools.chain.from_iterable(masked_values))
