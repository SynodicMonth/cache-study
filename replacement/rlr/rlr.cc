#include <cassert>
#include <map>
#include <vector>
#include "cache.h"

namespace {

/// Metadata for RLR cache line.
struct RlrLineMetadata {
  /// To estimate the reuse distance
  size_t age_counter = 0;
  /// Whether the previous access was not a prefetch
  bool type_register = false;
  /// If the cache line was reused.
  bool hit_register = false;
};

std::map<CACHE*, std::vector<RlrLineMetadata>> augmented_cache;
std::map<CACHE*, std::vector<uint8_t>> num_hits;
std::map<CACHE*, std::vector<uint8_t>> num_misses;
std::map<CACHE*, std::vector<size_t>> reuse_distances;
std::map<CACHE*, std::vector<size_t>> accumulators;

}  // namespace

void CACHE::initialize_replacement() {
  ::augmented_cache[this] = std::vector<RlrLineMetadata>(NUM_SET * NUM_WAY);
  ::num_hits[this] = std::vector<uint8_t>(NUM_SET);
  ::num_misses[this] = std::vector<uint8_t>(NUM_SET);
  ::reuse_distances[this] = std::vector<size_t>(NUM_SET);
  ::accumulators[this] = std::vector<size_t>(NUM_SET);
}

uint32_t CACHE::find_victim(
  uint32_t triggering_cpu,
  uint64_t instr_id,
  uint32_t set,
  const BLOCK* current_set,
  uint64_t ip,
  uint64_t full_addr,
  uint32_t type
) {
  auto begin = std::next(std::begin(::augmented_cache[this]), set * NUM_WAY);
  auto end = std::next(begin, NUM_WAY);

  size_t reuse_distance = ::reuse_distances[this].at(set);

  // get the minimum priority
  auto victim = std::min_element(
    begin, end,
    [&](const RlrLineMetadata& metadata1, const RlrLineMetadata& metadata2) {
      uint8_t priority1 = 0;
      priority1 += metadata1.age_counter > reuse_distance ? 0 : 8;
      priority1 += metadata1.type_register;
      priority1 += metadata1.hit_register;

      uint8_t priority2 = 0;
      priority2 += metadata2.age_counter > reuse_distance ? 0 : 8;
      priority2 += metadata2.type_register;
      priority2 += metadata2.hit_register;

      return priority1 < priority2 ||
             (priority1 == priority2 &&
              metadata1.age_counter < metadata2.age_counter);
    }
  );

  assert(begin <= victim);
  assert(victim < end);

  return static_cast<uint32_t>(std::distance(begin, victim));
}

void CACHE::update_replacement_state(
  uint32_t triggering_cpu,
  uint32_t set,
  uint32_t way,
  uint64_t full_addr,
  uint64_t ip,
  uint64_t victim_addr,
  uint32_t type,
  uint8_t hit
) {
  auto begin = std::next(std::begin(::augmented_cache[this]), set * NUM_WAY);
  auto end = std::next(begin, NUM_WAY);

  auto& metadata = ::augmented_cache[this].at(set * NUM_WAY + way);

  if (hit) {
    // 1. send the age_counter to accumulator
    // 2. reset the age_counter
    // 3. update the type_register
    // 4. update the hit_register
    // 5. accumulate num_hits
    // 6. if num_hits reaches 32, calculate reuse_distance

    ::accumulators[this].at(set) += metadata.age_counter;
    metadata.age_counter = 0;
    metadata.type_register = access_type{type} != access_type::PREFETCH;
    metadata.hit_register = true;

    ::num_hits[this].at(set) += 1;

    if (::num_hits[this].at(set) == 32) {
      // 2 * accumulator / num_hits(32)
      // shr by 5 and shl by 1, which is equivalent to shr by 4
      ::reuse_distances[this].at(set) = ::accumulators[this].at(set) >> 4;

      // reset the accumulator
      ::accumulators[this].at(set) = 0;

      // reset the num_hits
      ::num_hits[this].at(set) = 0;
    }
  } else {
    // 1. increase num_misses of the set
    // 2. reset the age_counter of the evicted cache line
    // 3. if num_misses is 8, reset the num_misses and increase the age_counter
    // in the set

    ::num_misses[this].at(set) += 1;

    if (::num_misses[this].at(set) == 8) {
      ::num_misses[this].at(set) = 0;

      for (auto it = begin; it != end; ++it) {
        it->age_counter += 1;
      }
    }

    metadata.age_counter = 0;
    metadata.type_register = access_type{type} != access_type::PREFETCH;
    metadata.hit_register = false;
  }
}

void CACHE::replacement_final_stats() {}