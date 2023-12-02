import unittest
from typing import Optional, Tuple


class CacheLine:
    def __init__(self) -> None:
        self.tag: int = 0
        self.valid: bool = False
        self.dirty: bool = False


class CacheSet:
    def __init__(self, associativity: int) -> None:
        self.cache_lines = [CacheLine() for _ in range(associativity)]

    def __getitem__(self, index: int) -> CacheLine:
        return self.cache_lines[index]

    def __setitem__(self, index: int, value: CacheLine) -> None:
        self.cache_lines[index] = value

    def __len__(self) -> int:
        return len(self.cache_lines)

    def __iter__(self):
        return iter(self.cache_lines)

    def reset(self):
        for line in self.cache_lines:
            line.tag = 0
            line.valid = False
            line.dirty = False


class Cache:
    def __init__(self, num_sets: int, associativity: int, block_size: int) -> None:
        self.cache_sets = [CacheSet(associativity) for _ in range(num_sets)]
        self.block_size = block_size

    def reset(self):
        for cache_set in self.cache_sets:
            cache_set.reset()

    def __getitem__(self, index: int) -> CacheSet:
        return self.cache_sets[index]

    def __setitem__(self, index: int, value: CacheSet) -> None:
        self.cache_sets[index] = value

    def __len__(self) -> int:
        return len(self.cache_sets)

    def set_index(self, address: int) -> int:
        return (address // self.block_size) % len(self.cache_sets)

    def tag(self, address: int) -> int:
        return address // self.block_size // len(self.cache_sets)

    def offset(self, address: int) -> int:
        return address % self.block_size

    def read(self, address: int) -> bool:
        """Perform read to the given address.
        """
        set_index = self.set_index(address)
        tag = self.tag(address)
        for line in self[set_index]:
            if line.valid and line.tag == tag:
                return True

        return False

    def write(self, address: int) -> bool:
        """Perform write to the given address.

            This will set the dirty flag of the cache line.
        """

        set_index = self.set_index(address)
        tag = self.tag(address)

        for line in self[set_index]:
            if line.valid and line.tag == tag:
                line.dirty = True
                return True

        return False

    def evict_at(self, set_index: int, i: int) -> Optional[Tuple[int, bool]]:
        """Evict a cache line by set and idx in set.

            Returns the evicted tag (if valid)
        """

        line = self[set_index][i]

        if line.valid:
            line.valid = False
            return line.tag, line.dirty
        else:
            return None

    def insert_at(self, set_index: int, i: int, tag: int) -> bool:
        line = self[set_index][i]

        if line.valid:
            return False

        else:
            line.tag = tag
            line.dirty = False
            line.valid = True
            return True

    def try_insert(self, address: int) -> bool:
        set_index = self.set_index(address)
        tag = self.tag(address)

        if self.read(address):
            return False

        for line in self[set_index]:
            if not line.valid:
                line.tag = tag
                line.dirty = False
                line.valid = True
                return True

        return False


class CacheTest(unittest.TestCase):
    def test_functionality(self):
        cache = Cache(2048, 16, 64)

        self.assertFalse(cache.read(0x00000000))
        self.assertFalse(cache.read(0x00000040))
        self.assertFalse(cache.read(0x00000080))
        self.assertFalse(cache.read(0x000000c0))

        self.assertTrue(cache.try_insert(0x00000000))
        self.assertTrue(cache.try_insert(0x00000040))
        self.assertTrue(cache.try_insert(0x00000080))
        self.assertTrue(cache.try_insert(0x000000c0))

        tag, dirty = cache.evict_at(0, 0)
        self.assertEqual(tag, 0)
        self.assertFalse(dirty)

        self.assertTrue(cache.try_insert(0x00000000))

        self.assertEqual(cache.set_index(0x00000040), 1)
        self.assertEqual(cache.tag(0x00000040), 0)

        self.assertFalse(cache.try_insert(0x00000040))
        self.assertFalse(cache.try_insert(0x00000080))
        self.assertFalse(cache.try_insert(0x000000c0))

        self.assertTrue(cache.read(0x00000000))
        self.assertTrue(cache.read(0x00000040))
        self.assertTrue(cache.read(0x00000080))
        self.assertTrue(cache.read(0x000000c0))

        self.assertTrue(cache.write(0x00000000))
        self.assertTrue(cache.write(0x00000040))
        self.assertTrue(cache.write(0x00000080))
        self.assertTrue(cache.write(0x000000c0))

        tag, dirty = cache.evict_at(0, 0)
        self.assertEqual(tag, 0)
        self.assertTrue(dirty)

        cache.reset()

        # test multiple tags in a set
        self.assertTrue(cache.try_insert(0x00000040))
        self.assertTrue(cache.try_insert(0x01000040))
        self.assertTrue(cache.try_insert(0x02000040))
        self.assertTrue(cache.try_insert(0x03000040))
        self.assertTrue(cache.try_insert(0x04000040))
        self.assertTrue(cache.try_insert(0x05000040))
        self.assertTrue(cache.try_insert(0x06000040))
        self.assertTrue(cache.try_insert(0x07000040))
        self.assertTrue(cache.try_insert(0x08000040))
        self.assertTrue(cache.try_insert(0x09000040))
        self.assertTrue(cache.try_insert(0x0a000040))
        self.assertTrue(cache.try_insert(0x0b000040))
        self.assertTrue(cache.try_insert(0x0c000040))
        self.assertTrue(cache.try_insert(0x0d000040))
        self.assertTrue(cache.try_insert(0x0e000040))
        self.assertTrue(cache.try_insert(0x0f000040))

        self.assertFalse(cache.try_insert(0x10000040))

        tag, dirty = cache.evict_at(1, 3)
        self.assertEqual(tag, cache.tag(0x03000040))
        self.assertFalse(dirty)

        self.assertTrue(cache.try_insert(0x10000040))


if __name__ == "__main__":
    unittest.main()
