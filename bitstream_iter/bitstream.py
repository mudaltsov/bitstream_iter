import itertools


class BitStream(object):
    """Bit stream that uses iterators for reading and writing."""

    LSB_FIRST = tuple((1 << n) for n in range(8))
    MSB_FIRST = tuple(reversed(LSB_FIRST))

    def __init__(self, bits=None):
        """Initialize a new bit stream.

        Args:
            bits (iterable, optional):
                Initial bit values to write to the stream.
                Each value is converted to 0 or 1 by truth value testing.
        """
        self._bit_sources = []

        if bits is not None:
            self.write_bits(bits=bits)

    @classmethod
    def from_bytes(cls, bytes, bitmasks=MSB_FIRST):
        """Create a bit stream containing bytes.

        Args:
            bytes (iterable):
                Initial byte values to write to the stream.
                Each value must be an integer.

            bitmasks (sequence, optional):
                Bitwise masks to AND with each byte to generate bits.
                Default is all 8 bits with most significant bit first.

        Returns:
            A new BitStream initialized with the given bytes.
        """
        stream = cls()
        stream.write_bytes(bytes=bytes, bitmasks=bitmasks)
        return stream

    def __iter__(self):
        """Implements iter(self).

        Returns:
            Iterator that produces 0 or 1 for each bit removed from the
            start of the stream, until all bits have been exhausted.
        """
        return itertools.chain.from_iterable(self._bit_sources)

    def bytes(self, bitmasks=MSB_FIRST):
        """Generate bytes from the bit stream.

        Args:
            bitmasks (sequence, optional):
                Bitwise masks to OR together for each bit to generate bytes.
                Default is all 8 bits with most significant bit first.

        Yields:
            Integers formed by bits removed from the start of the stream.
            Bits that don't form a complete byte will remain in the stream.
        """
        while True:
            bits = tuple(itertools.islice(self, len(bitmasks)))

            if len(bits) < len(bitmasks):
                if len(bits):
                    self.write_bits(bits)
                return

            result = 0
            for mask, bit in zip(bitmasks, bits):
                result |= mask * bit

            yield result

    def write_bits(self, bits):
        """Write bits to the end of the stream.

        Args:
            bits (iterable):
                Bit values to write. Each value is converted to 0 or 1
                by truth value testing.
        """
        self._bit_sources.append(int(bool(b)) for b in bits)

    def write_bytes(self, bytes, bitmasks=MSB_FIRST):
        """Write bytes to the end of the stream.

        Args:
            bytes (iterable):
                Byte values to write. Each value must be an integer.

            masks (sequence, optional):
                Bitwise masks to AND with each byte to generate bits.
                Default is all 8 bits with most significant bit first.
        """
        bits = (int(bool(byte & mask)) for byte in bytes for mask in bitmasks)
        self._bit_sources.append(bits)
