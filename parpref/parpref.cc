#include "cache.h"
#include "par/parhelper.h"
#include <iostream>
#include <bitset>
#include <map>
#include <vector>

namespace
{
constexpr std::size_t REGION_COUNT = 128;
constexpr int MAX_DISTANCE = 256;
constexpr int PREFETCH_DEGREE = 2;

struct region_type {
  uint64_t vpn;
  std::bitset<PAGE_SIZE / BLOCK_SIZE> access_map{};
  std::bitset<PAGE_SIZE / BLOCK_SIZE> prefetch_map{};
  uint64_t lru;

  static uint64_t region_lru;

  region_type() : region_type(0) {}
  explicit region_type(uint64_t allocate_vpn) : vpn(allocate_vpn), lru(region_lru++) {}
};
uint64_t region_type::region_lru = 0;

std::map<CACHE*, std::array<region_type, REGION_COUNT>> regions;

auto page_and_offset(uint64_t addr)
{
  auto page_number = addr >> LOG2_PAGE_SIZE;
  auto page_offset = (addr & champsim::msl::bitmask(LOG2_PAGE_SIZE)) >> LOG2_BLOCK_SIZE;
  return std::pair{page_number, page_offset};
}

bool check_cl_access(CACHE* cache, uint64_t v_addr)
{
  auto [vpn, page_offset] = page_and_offset(v_addr);
  auto region = std::find_if(std::begin(regions.at(cache)), std::end(regions.at(cache)), [vpn = vpn](auto x) { return x.vpn == vpn; });

  return (region != std::end(regions.at(cache))) && region->access_map.test(page_offset);
}

bool check_cl_prefetch(CACHE* cache, uint64_t v_addr)
{
  auto [vpn, page_offset] = page_and_offset(v_addr);
  auto region = std::find_if(std::begin(regions.at(cache)), std::end(regions.at(cache)), [vpn = vpn](auto x) { return x.vpn == vpn; });

  return (region != std::end(regions.at(cache))) && region->prefetch_map.test(page_offset);
}

} // anonymous namespace

void CACHE::prefetcher_initialize() {
  regions.insert_or_assign(this, decltype(regions)::mapped_type{});
}

uint32_t CACHE::prefetcher_cache_operate(uint64_t addr, uint64_t ip, uint8_t cache_hit, bool useful_prefetch, uint8_t type, uint32_t metadata_in)
{
  // uint64_t pf_addr = addr + (1 << LOG2_BLOCK_SIZE);
  // prefetch_line(pf_addr, true, metadata_in);

  // print out the par::estimated_reuse_time_evicted map
  // for (auto& [_cache , _map] : par::estimated_reuse_time_evicted) {
  //   std::cout << "cache: " << _cache << std::endl;
  //   for (auto& [block_addr, reuse_time] : _map) {
  //     std::cout << block_addr << " " << reuse_time << std::endl;
  //   }
  // }

  // for (auto& [block_addr, reuse_time] : par::estimated_reuse_time_evicted[this]) {
  //   std::cout << block_addr << " " << reuse_time << std::endl;
  // }
  
  // clear all the overdue estimated reuse time
  for (auto it = par::estimated_reuse_time_evicted[this].begin(); it != par::estimated_reuse_time_evicted[this].end();) {
    if (it->second < current_cycle) {
      it = par::estimated_reuse_time_evicted[this].erase(it);
    } else {
      ++it;
    }
  }

  // prefetch the earliest estimated reuse block
  if (par::estimated_reuse_time_evicted[this].size() > 0) {
    auto earliest_reuse_block = std::min_element(
      std::begin(par::estimated_reuse_time_evicted[this]), 
      std::end(par::estimated_reuse_time_evicted[this]), 
      [](auto a, auto b) { return a.second < b.second; });

    // // see if the earliest reuse block is younger than any of the blocks in the cache
    // // if so, prefetch it
    // if (earliest_reuse_block->second > current_cycle) {
    //   auto pf_addr = earliest_reuse_block->first * BLOCK_SIZE;
    //   if (get_line_state(pf_addr) == LINE_EMPTY) {
    //     prefetch_line(pf_addr, true, metadata_in);
    //   }
    // }


    // std::cout << earliest_reuse_block->second << std::endl;
    // if its after the current cycle, prefetch it
    auto pf_addr = earliest_reuse_block->first * BLOCK_SIZE;

    prefetch_line(pf_addr, true, metadata_in);
    // std::cout << "prefetching " << earliest_reuse_block->first << " at " << current_cycle << std::endl;

    par::estimated_reuse_time_evicted[this].erase(earliest_reuse_block);
  }
  
  auto [current_vpn, page_offset] = ::page_and_offset(addr);
  auto demand_region = std::find_if(std::begin(::regions.at(this)), std::end(::regions.at(this)), [vpn = current_vpn](auto x) { return x.vpn == vpn; });

  if (demand_region == std::end(::regions.at(this))) {
    // not tracking this region yet, so replace the LRU region
    demand_region = std::min_element(std::begin(::regions.at(this)), std::end(::regions.at(this)), [](auto x, auto y) { return x.lru < y.lru; });
    *demand_region = region_type{current_vpn};
    return metadata_in;
  }

  // mark this demand access
  demand_region->access_map.set(page_offset);

  // attempt to prefetch in the positive, then negative direction
  for (auto direction : {1, -1}) {
    for (int i = 1, prefetches_issued = 0; i <= MAX_DISTANCE && prefetches_issued < PREFETCH_DEGREE; i++) {
      const auto pos_step_addr = addr + direction * (i * (signed)BLOCK_SIZE);
      const auto neg_step_addr = addr - direction * (i * (signed)BLOCK_SIZE);
      const auto neg_2step_addr = addr - direction * (2 * i * (signed)BLOCK_SIZE);

      if (::check_cl_access(this, neg_step_addr) && ::check_cl_access(this, neg_2step_addr) && !::check_cl_access(this, pos_step_addr)
          && !::check_cl_prefetch(this, pos_step_addr)) {
        // found something that we should prefetch
        if ((addr >> LOG2_BLOCK_SIZE) != (pos_step_addr >> LOG2_BLOCK_SIZE)) {
          bool prefetch_success = prefetch_line(pos_step_addr, (get_mshr_occupancy_ratio() < 0.5), metadata_in);
          if (prefetch_success) {
            auto [pf_vpn, pf_page_offset] = ::page_and_offset(pos_step_addr);
            auto pf_region = std::find_if(std::begin(::regions.at(this)), std::end(::regions.at(this)), [vpn = pf_vpn](auto x) { return x.vpn == vpn; });

            if (pf_region == std::end(::regions.at(this))) {
              // we're not currently tracking this region, so allocate a new region so we can mark it
              pf_region = std::min_element(std::begin(::regions.at(this)), std::end(::regions.at(this)), [](auto x, auto y) { return x.lru < y.lru; });
              *pf_region = region_type{pf_vpn};
            }

            pf_region->prefetch_map.set(pf_page_offset);
            prefetches_issued++;
          }
        }
      }
    }
  }


  return metadata_in;
}

uint32_t CACHE::prefetcher_cache_fill(uint64_t addr, uint32_t set, uint32_t way, uint8_t prefetch, uint64_t evicted_addr, uint32_t metadata_in)
{
  return metadata_in;
}

void CACHE::prefetcher_cycle_operate() {}

void CACHE::prefetcher_final_stats() {}
