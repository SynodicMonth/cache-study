#ifndef PAR_H
#define PAR_H

#include <algorithm>
#include <cassert>
#include <map>
#include <vector>

#include "cache.h"

// namespace par
// {
static std::map<CACHE*, std::vector<uint64_t>> last_used_cycles;
static std::map<CACHE*, std::vector<uint64_t>> total_set_accesses;
static std::map<CACHE*, std::map<uint64_t, uint64_t>> estimated_reuse_time; // <cache*, <block_addr, reuse_time>>
// }

#endif // PAR_H