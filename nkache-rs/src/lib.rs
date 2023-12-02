/// A cache line/block.
///
/// This does not contain the actual data, but rather the metadata
#[derive(Debug, Clone)]
pub struct CacheLine {
    pub tag: u64,
    pub valid: bool,
    pub dirty: bool,
}

impl Default for CacheLine {
    fn default() -> Self {
        CacheLine {
            tag: 0,
            valid: false,
            dirty: false,
        }
    }
}

/// A cache set of N lines.
///
/// N is the associativity of the cache.
pub struct CacheSet<const N: usize> {
    pub lines: [CacheLine; N],
}

impl<const N: usize> Default for CacheSet<N> {
    fn default() -> Self {
        let lines = std::array::from_fn(|_| CacheLine::default());
        CacheSet { lines }
    }
}

impl<const N: usize> CacheSet<N> {
    pub fn reset(&mut self) {
        for line in self.lines.iter_mut() {
            line.valid = false;
            line.dirty = false;
        }
    }
}

/// The type of access to the cache.
pub enum AccessType {
    Load,
    /// Read for Ownership
    Rfo,
    Prefetch,
    Write,
    Translation,
}

/// A trace entry for a cache access.
pub struct TraceEntry {
    /// The type of access to the cache.
    pub access_type: AccessType,
    /// The address of the access.
    pub address: u64,
}

/// A cache with M sets of N lines.
///
/// M is the number of sets in the cache.
/// N is the associativity of the cache.
pub struct Cache<const M: usize, const N: usize> {
    pub sets: [CacheSet<N>; M],
    pub block_size: usize,
}

impl<const M: usize, const N: usize> Cache<M, N> {
    pub fn new(block_size: usize) -> Self {
        let sets = std::array::from_fn(|_| CacheSet::default());
        Cache { sets, block_size }
    }

    pub fn reset(&mut self) {
        for set in self.sets.iter_mut() {
            set.reset();
        }
    }

    fn set_index(&self, address: u64) -> usize {
        let mask = (M - 1) as u64;
        (address >> self.block_size.trailing_zeros()) as usize & mask as usize
    }

    fn tag(&self, address: u64) -> u64 {
        address >> (self.block_size.trailing_zeros() + (M - 1).trailing_zeros() + 1)
    }

    pub fn offset(&self, address: u64) -> usize {
        (address & (self.block_size as u64 - 1)) as usize
    }

    /// Perform read operation on the cache.
    /// 
    /// Returns true if the read was a hit, false otherwise.
    pub fn read(&self, address: u64) -> bool {
        let set_index = self.set_index(address);
        let tag = self.tag(address);

        for line in self.sets[set_index].lines.iter() {
            if line.valid && line.tag == tag {
                return true;
            }
        }
        false
    }

    /// Perform write operation on the cache.
    /// 
    /// Returns true if the write was a hit, false otherwise.
    pub fn write(&mut self, address: u64) -> bool {
        let set_index = self.set_index(address);
        let tag = self.tag(address);

        for line in self.sets[set_index].lines.iter_mut() {
            if line.valid && line.tag == tag {
                line.dirty = true;
                return true;
            }
        }
        false
    }

    /// Evict a cache line from the cache.
    /// 
    /// Returns the tag of the evicted line.
    pub fn evict_at(&mut self, set_index: usize, line_index: usize) -> Option<u64> {
        let line = &mut self.sets[set_index].lines[line_index];
        if line.valid {
            line.valid = false;
            line.dirty = false;
            Some(line.tag)
        } else {
            None
        }
    }

    /// Insert a cache line into the cache.
    /// 
    /// Returns if the insertion was successful.
    pub fn insert_at(&mut self, set_index: usize, line_index: usize, tag: u64) -> bool {
        let line = &mut self.sets[set_index].lines[line_index];
        if !line.valid {
            line.valid = true;
            line.dirty = false;
            line.tag = tag;
            true
        } else {
            false
        }
    }
}
