import itertools
from collections import deque
from typing import Any, Iterable, Iterator, MutableSequence, NewType, Sequence

from bitstream_iter import bitmasks


Bit = NewType('Bit', int)


def make_bit(value: Any) -> Bit:
    """Convert any value to 0 or 1 using truth value testing."""
    if value:
        return Bit(1)
    return Bit(0)


class BitStream:
    """Binary stream that uses iterators for I/O with bit granularity.

    BitStream()                   -> empty BitStream
    BitStream(bits)               -> BitStream with bit values
    BitStream.from_bytes(values)  -> BitStream with bits extracted from bytes

    The constructors take two optional arguments:
        byte_bitmasks - bit masks for converting bytes
        end_fill - bit value to fill the end of the stream for bytes/ints

    I/O methods with b = BitStream(...):
        Write data with b.write_bits(), b.write_bytes(), or b.write_ints().
        Read data with iter(b), bytes(b), b.iter_bytes(), or b.iter_ints().
    """

    def __init__(self, bits: Iterable[Any] = None,
                 byte_bitmasks: Sequence[int] = None,
                 end_fill: Any = 0) -> None:
        """Initialize a new bit stream.

        Args:
            bits: Initial bit values to lazily write to the stream.
                Each value is converted to 0 or 1 by truth value testing.

            byte_bitmasks: Bit masks used for byte <-> bit conversion.
                Default is all 8 bits with most significant bit first.

            end_fill: Bit value to fill the end of the stream for generating
                bytes/ints. Default is 0 to consume all bits from the stream.
                None leaves unused bits in the stream, and requires a new
                iterator to read data after adding more bits.
        """
        self.byte_bitmasks = byte_bitmasks or bitmasks.UINT8_MSB_FIRST
        self.end_fill = None if end_fill is None else Bit(end_fill)

        self._bit_sources = deque()  # type: MutableSequence[Iterator[Bit]]

        if bits is not None:
            self.write_bits(bits)

    @classmethod
    def from_bytes(cls, values: Iterable[int],
                   byte_bitmasks: Sequence[int] = None,
                   end_fill: Any = 0) -> 'BitStream':
        """Create a bit stream containing bytes.

        Args:
            values: Initial byte values to lazily write to the stream.

            byte_bitmasks: Bit masks used for byte <-> bit conversion.
                Default is all 8 bits with most significant bit first.

            end_fill: Bit value to fill the end of the stream for generating
                bytes/ints. Default is 0 to consume all bits from the stream.
                None leaves unused bits in the stream, and requires a new
                iterator to read data after adding more bits.

        Returns:
            A new BitStream initialized with the given data.
        """
        stream = cls(byte_bitmasks=byte_bitmasks, end_fill=end_fill)
        stream.write_bytes(values=values)
        return stream

    def __bytes__(self) -> bytes:
        """Implements bytes(self).

        Returns:
            Bytes formed by consuming all possible bits from the stream.
            Uses byte_bitmasks and end_fill from the constructor.
        """
        return bytes(self.iter_bytes())

    def __iter__(self) -> Iterator[Bit]:
        """Implements iter(self).

        Yields:
            Next bit value as integer 0 or 1, consumed from the start of the
            stream until all all bits have been exhausted. After the end of
            the stream is reached, any new data written to the stream is only
            accessible by creating a new iterator.
        """
        while len(self._bit_sources):
            bit = next(self._bit_sources[0], None)
            if bit is None:
                del self._bit_sources[0]
                continue

            yield bit

    def iter_bytes(self, masks: Sequence[int] = None,
                   end_fill: Any = -1) -> Iterator[int]:
        """Iterator that generates bytes from the bit stream.

        Args:
            masks: Bit masks used for combining bits into a byte.
                Default is byte_bitmasks specified in the constructor.

            end_fill: Bit value to fill the end of the stream for generating
                bytes. Default or -1 uses end_fill from the constructor.
                None leaves unused bits in the stream.

        Returns:
            Iterator that produces each byte by consuming bits from the start
            of the stream, until all bits have been exhausted. After the end
            of the stream is reached, any new data written to the stream is
            only accessible by creating a new iterator.

        """
        masks = masks or self.byte_bitmasks
        return self.iter_ints(masks=masks, end_fill=end_fill)

    def iter_ints(self, masks: Sequence[int],
                  end_fill: Any = -1) -> Iterator[int]:
        """Iterator that generates integers from the bit stream.

        Args:
            masks: Bit masks used for combining bits into an integer.

            end_fill: Bit value to fill the end of the stream for generating
                integers. Default or -1 uses end_fill from the constructor.
                None leaves unused bits in the stream.

        Yields:
            Next integer value formed by consuming bits from the start of
            the stream, until all bits have been exhausted. After the end
            of the stream is reached, any new data written to the stream is
            only accessible by creating a new iterator.
        """
        if end_fill == -1:
            end_fill = self.end_fill

        bit_iter = iter(self)

        while True:
            bits = list(itertools.islice(bit_iter, len(masks)))

            if not len(bits):
                return
            elif len(bits) < len(masks) and end_fill is None:
                self.write_bits(bits)
                return

            while end_fill and len(bits) < len(masks):
                bits.append(end_fill)

            yield bitmasks.combine(masks=masks, bits=bits)

    def write_bits(self, values: Iterable[Any],
                   at_start: bool = False) -> None:
        """Write lazily accessed bits at the end or start of the stream.

        Args:
            values: Bit values that are accessed when consuming stream bits.
                Each value is converted to 0 or 1 by truth value testing.

            at_start: Write bits at the start of the stream instead of the end.
        """
        bits = (make_bit(b) for b in values)
        if at_start:
            self._bit_sources.appendleft(bits)  # type: ignore # Deque in 3.6.1
        else:
            self._bit_sources.append(bits)

    def write_bytes(self, values: Iterable[int], masks: Sequence[int] = None,
                    at_start: bool = False) -> None:
        """Write lazily accessed bytes at the end or start of the stream.

        Args:
            values: Byte values that are accessed when consuming stream bits.

            masks: Bit masks applied to each byte to produce bits.
                Default is byte_bitmasks specified in the constructor.

            at_start: Write bits at the start of the stream instead of the end.
        """
        masks = masks or self.byte_bitmasks
        self.write_ints(values=values, masks=masks, at_start=at_start)

    def write_ints(self, values: Iterable[int], masks: Sequence[int],
                   at_start: bool = False) -> None:
        """Write lazily accessed integers at the end or start of the stream.

        Args:
            values: Values that are accessed when consuming stream bits.

            masks: Bit masks applied to each integer to produce bits.

            at_start: Write bits at the start of the stream instead of the end.
        """
        masked_values = (bitmasks.apply(masks=masks, value=v) for v in values)
        flat_values = itertools.chain.from_iterable(masked_values)

        self.write_bits(values=flat_values, at_start=at_start)
