#include "par/parhelper.h"
#include "cache.h"
#include <iostream>

namespace par {
    std::map<CACHE*, std::vector<uint64_t>> last_used_cycles;
    std::map<CACHE*, std::vector<uint64_t>> total_set_accesses;
    std::map<CACHE*, std::map<uint64_t, uint64_t>> estimated_reuse_time_in_cache;
    std::map<CACHE*, std::map<uint64_t, uint64_t>> estimated_reuse_time_evicted;
}

void CACHE::initialize_replacement() { 
  par::last_used_cycles[this] = std::vector<uint64_t>(NUM_SET * NUM_WAY); 
  par::total_set_accesses[this] = std::vector<uint64_t>(NUM_SET);
}

uint32_t CACHE::find_victim(uint32_t triggering_cpu, uint64_t instr_id, uint32_t set, const BLOCK* current_set, uint64_t ip, uint64_t full_addr, uint32_t type)
{
  auto begin = std::next(std::begin(par::last_used_cycles[this]), set * NUM_WAY);
  auto end = std::next(begin, NUM_WAY);

  // Find the way whose last use cycle is most distant
  auto victim = std::min_element(begin, end);
  assert(begin <= victim);
  assert(victim < end);

  // mave the victim to estimated_reuse_time_evicted
  auto victim_addr = current_set[std::distance(begin, victim)].address;
  // std::cout << "evicted" << victim_addr << " at " << current_cycle << std::endl;
  auto victim_reuse_time = par::estimated_reuse_time_in_cache[this][victim_addr / BLOCK_SIZE];
  if (victim_reuse_time > current_cycle) {
    par::estimated_reuse_time_evicted[this][victim_addr / BLOCK_SIZE] = victim_reuse_time;
    // std::cout << "evict " << victim_addr / BLOCK_SIZE << " " << victim_reuse_time << " at " << current_cycle << std::endl;
  }
  return static_cast<uint32_t>(std::distance(begin, victim)); // cast protected by prior asserts
}

void CACHE::update_replacement_state(uint32_t triggering_cpu, uint32_t set, uint32_t way, uint64_t full_addr, uint64_t ip, uint64_t victim_addr, uint32_t type,
                                     uint8_t hit)
{
  // Mark the way as being used on the current cycle
  if (!hit || access_type{type} != access_type::WRITE) {
    par::last_used_cycles[this].at(set * NUM_WAY + way) = current_cycle;
    // par::total_set_accesses[this].at(set)++;
  } // Skip this for writeback hits

  // Update the estimated reuse time for the victim block
  if (hit) {
    if (par::last_used_cycles[this].at(set * NUM_WAY + way) != 0) {
      auto reuse_time = (current_cycle - par::last_used_cycles[this].at(set * NUM_WAY + way)) * 2 + current_cycle;
      par::estimated_reuse_time_in_cache[this][full_addr / BLOCK_SIZE] = reuse_time;
      // std::cout << "add " << full_addr / BLOCK_SIZE << " " << reuse_time << std::endl;
    }
  }
}

void CACHE::replacement_final_stats() {}
