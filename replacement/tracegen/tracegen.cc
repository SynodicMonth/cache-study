#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>

#include "cache.h"

namespace {
std::stringstream llc_access_traces;
std::map<CACHE*, std::vector<uint64_t>> last_used_cycles;
}  // namespace

void CACHE::initialize_replacement() {
  ::last_used_cycles[this] = std::vector<uint64_t>(NUM_SET * NUM_WAY);
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
  auto begin = std::next(std::begin(::last_used_cycles[this]), set * NUM_WAY);
  auto end = std::next(begin, NUM_WAY);

  // Find the way whose last use cycle is most distant
  auto victim = std::min_element(begin, end);
  assert(begin <= victim);
  assert(victim < end);
  return static_cast<uint32_t>(std::distance(begin, victim)
  );  // cast protected by prior asserts
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
  // Mark the way as being used on the current cycle
  if (!hit || access_type{type} != access_type::WRITE)  // Skip this for
                                                        // writeback hits
    ::last_used_cycles[this].at(set * NUM_WAY + way) = current_cycle;

  // Write to llc_access_traces
  // <Access Type, Address>

  // Access Type
  switch (access_type{type}) {
    case access_type::LOAD:
      llc_access_traces << "l ";
      break;
    case access_type::RFO:
      llc_access_traces << "r ";
      break;
    case access_type::PREFETCH:
      llc_access_traces << "p ";
      break;
    case access_type::WRITE:
      llc_access_traces << "w ";
      break;
    case access_type::TRANSLATION:
      llc_access_traces << "t ";
      break;
    default:
      assert(false);
  }

  // Address
  llc_access_traces << std::hex << full_addr << std::endl;
}

void CACHE::replacement_final_stats() {
  // write llc trace to file
  std::ofstream llc_access_trace_file;
  llc_access_trace_file.open("llc_trace.txt");
  llc_access_trace_file << llc_access_traces.str();
  llc_access_trace_file.close();
}
