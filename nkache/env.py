from nkache.cache import Cache
from typing import Dict, Tuple, Optional, List
from collections import deque
from bisect import bisect_right

import numpy as np
import numpy.typing as npt


class CacheEnv:
    def __init__(self, num_sets: int, associativity: int, block_size: int) -> None:
        self.cache = Cache(num_sets, associativity, block_size)

        self.associativity = associativity

        # total accesses
        self.total_access_cnt = 0
        # total misses
        self.total_miss_cnt = 0

        # set accesses
        self.set_access_cnts = [0 for _ in range(num_sets)]

        # set accesses at the last miss
        self.set_access_cnts_at_miss = [0 for _ in range(num_sets)]

        # set accesses at the last access to a given address
        self.address_preuses: Dict[int, int] = {}

        # set accesses at the last access to the given line (set_index, tag)
        self.line_preuses_0: Dict[Tuple[int, int], int] = {}

        # set accesses at the last-last access to the given line (set_index, tag)
        self.line_preuses_1: Dict[Tuple[int, int], int] = {}

        # set accesses at the insert of the given line (set_index, tag)
        self.line_insertion_set_access_cnts: Dict[Tuple[int, int], int] = {}

        # line last access type
        self.line_last_access_type: Dict[Tuple[int, int], Optional[str]] = {}

        # access count of different types
        self.line_access_cnts: Dict[Tuple[int, int], Dict[str, int]] = {}

        # line hits since insertion
        self.line_hits_since_insertion: Dict[Tuple[int, int], int] = {}

        # line recency
        self.line_recency: Dict[int, deque[Tuple[int, int]]] = {}

        # traces
        self.traces: List[Tuple[str, int]] = []

        # belady list
        self.belady_list: Dict[Tuple[int, int], List[int]] = {}

    def observation_space(self) -> int:
        return 9 + 16 * self.associativity

    def action_space(self) -> int:
        return self.associativity

    def prepare_belady(self) -> None:
        self.belady_list = {}

        for idx, (_, addr) in enumerate(self.traces):
            set_index = self.cache.set_index(addr)
            tag = self.cache.tag(addr)
            if (set_index, tag) not in self.belady_list:
                self.belady_list[(set_index, tag)] = []

            self.belady_list[(set_index, tag)].append(idx)

    def belady_replacement_optimal(self, trace_idx: int) -> Optional[List[int]]:
        addr = self.traces[trace_idx][1]
        set_index = self.cache.set_index(addr)

        tags = [line.tag if line.valid else None for line in self.cache[set_index]]

        if all(tag is None for tag in tags):
            return None

        next_access_indices = []

        for tag in tags:
            if tag is None:
                next_access_indices.append(-1)
                continue

            line_access_indices = self.belady_list[(set_index, tag)]
            next_access_idx = bisect_right(line_access_indices, trace_idx)

            if next_access_idx < len(line_access_indices):
                next_access_indices.append(
                    line_access_indices[next_access_idx])
            else:
                next_access_indices.append(float('inf'))

        return next_access_indices

    def update_set_recency(self, set_index: int, tag: int) -> None:
        if set_index not in self.line_recency:
            self.line_recency[set_index] = deque()

        if (set_index, tag) in self.line_recency[set_index]:
            self.line_recency[set_index].remove((set_index, tag))

        self.line_recency[set_index].append((set_index, tag))

    def load_trace(self, filename: str) -> None:
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if line == '':
                    continue

                access_type, addr = line.split()
                addr = int(addr, 16)

                self.traces.append((access_type, addr))

    def observe(self, trace_idx: int) -> npt.NDArray:
        observation: List[float] = []

        access_type, addr = self.traces[trace_idx]

        set_index = self.cache.set_index(addr)

        total_access_cnt = len(self.traces)

        # address preuse
        observation.append((
            self.set_access_cnts[set_index] -
            self.address_preuses.get(addr, 0)) / total_access_cnt
        )

        # one hot of last access type
        access_types = ['l', 'r', 'w', 'p', 't']
        observation.extend(
            [1 if access_type == at else 0 for at in access_types]
        )

        # set index
        observation.append(set_index / len(self.cache))

        # set accesses total
        observation.append(self.set_access_cnts[set_index] / total_access_cnt)

        # set accesses since miss
        observation.append((
            self.set_access_cnts[set_index] -
            self.set_access_cnts_at_miss[set_index]) / total_access_cnt
        )

        line_observations: List[List[int]] = [[]
                                              for _ in range(self.associativity)]

        # line informations
        curr_line_idx = 0
        for line in self.cache[set_index]:
            # dirty
            line_observations[curr_line_idx].append(1 if line.dirty else 0)
            # preuse
            preuse_0 = self.line_preuses_0.get((set_index, line.tag), 0)
            preuse_1 = self.line_preuses_1.get((set_index, line.tag), 0)
            preuse = preuse_0 - preuse_1
            line_observations[curr_line_idx].append(preuse)
            # age since insertion
            age = self.set_access_cnts[set_index] - \
                self.line_insertion_set_access_cnts.get(
                    (set_index, line.tag), 0)
            line_observations[curr_line_idx].append(age)
            # age since last access
            age = self.set_access_cnts[set_index] - preuse_0
            line_observations[curr_line_idx].append(age)
            # one hot of last access type
            last_access_type = self.line_last_access_type[(
                set_index, line.tag)]
            line_observations[curr_line_idx].extend(
                [1 if last_access_type == at else 0 for at in access_types]
            )
            # access count of different types
            for at in access_types:
                line_observations[curr_line_idx].append(
                    self.line_access_cnts[(set_index, line.tag)].get(at, 0)
                )
            # hits since insertion
            line_observations[curr_line_idx].append(
                self.line_hits_since_insertion.get((set_index, line.tag), 0)
            )
            # recency
            recency = self.line_recency[set_index].index((set_index, line.tag))
            line_observations[curr_line_idx].append(recency)

            curr_line_idx += 1
            

        # normalize of line observations into [0, 1]
        line_observations: npt.NDArray = np.array(
            line_observations, dtype=np.float32)

        min_line_observations = np.min(line_observations, axis=0)
        max_line_observations = np.max(line_observations, axis=0)
        range_line_observations = max_line_observations - min_line_observations
        # normalize, if range is 0, then set to 0
        line_observations = ((line_observations - min_line_observations) /
                             np.where(range_line_observations == 0,
                                      np.where(min_line_observations == 0, 1, min_line_observations), range_line_observations))

        # flatten line observations
        line_observations: List[float] = line_observations.flatten().tolist()

        observation.extend(line_observations)
        observation: npt.NDArray = np.array(observation, dtype=np.float32)

        return observation

    def execute_single_trace(self, trace_idx: int) -> bool:
        access_type, addr = self.traces[trace_idx]
        set_index = self.cache.set_index(addr)
        tag = self.cache.tag(addr)

        self.total_access_cnt += 1
        self.set_access_cnts[set_index] += 1

        success = False
        non_compulsory_miss = False

        if access_type == 'w':
            success = self.cache.write(addr)
        else:
            success = self.cache.read(addr)

        if not success:
            self.total_miss_cnt += 1
            self.set_access_cnts_at_miss[set_index] = self.set_access_cnts[set_index]
            success = self.cache.try_insert(addr)

            if success:
                # compulsory miss
                self.line_preuses_0[(set_index, tag)
                                    ] = self.set_access_cnts[set_index]
                self.line_preuses_1.pop((set_index, tag), None)

                self.line_insertion_set_access_cnts[(
                    set_index, tag)] = self.set_access_cnts[set_index]
                self.line_last_access_type[(set_index, tag)] = access_type

                self.line_access_cnts[(set_index, tag)] = {
                    at: 0 for at in ['l', 'r', 'w', 'p', 't']}
                self.line_access_cnts[(set_index, tag)][access_type] = 1

                self.line_hits_since_insertion[(set_index, tag)] = 0

                self.update_set_recency(set_index, tag)

            else:
                # non-compulsory miss
                non_compulsory_miss = True

        else:
            self.line_preuses_1[(set_index, tag)
                                ] = self.line_preuses_0[(set_index, tag)]
            self.line_preuses_0[(set_index, tag)
                                ] = self.set_access_cnts[set_index]

            self.line_last_access_type[(set_index, tag)] = access_type
            self.line_access_cnts[(set_index, tag)][access_type] += 1

            self.line_hits_since_insertion[(set_index, tag)] += 1

            self.update_set_recency(set_index, tag)

        self.address_preuses[addr] = self.set_access_cnts[set_index]

        return non_compulsory_miss

    def execute_till_non_compulsory_miss(self, trace_idx: int) -> Optional[int]:
        while trace_idx < len(self.traces):
            non_compulsory_miss = self.execute_single_trace(trace_idx)
            if non_compulsory_miss:
                return trace_idx
            trace_idx += 1

        return None

    def execute_action_on_non_compulsory_miss(self, trace_idx: int, action: int) -> bool:
        access_type, addr = self.traces[trace_idx]

        set_index = self.cache.set_index(addr)
        tag = self.cache.tag(addr)

        evict_result = self.cache.evict_at(set_index, action)

        if evict_result is None:
            return False

        evicted_tag, dirty = evict_result

        # clean the evicted line
        self.line_preuses_0.pop((set_index, evicted_tag), None)
        self.line_preuses_1.pop((set_index, evicted_tag), None)

        self.line_insertion_set_access_cnts.pop((set_index, evicted_tag), None)
        self.line_last_access_type.pop((set_index, evicted_tag), None)

        self.line_access_cnts.pop((set_index, evicted_tag), None)
        self.line_hits_since_insertion.pop((set_index, evicted_tag), None)

        self.line_recency[set_index].remove((set_index, evicted_tag))

        success = self.cache.insert_at(set_index, action, tag)

        if not success:
            return False

        self.line_preuses_0[(set_index, tag)] = self.set_access_cnts[set_index]
        self.line_preuses_1.pop((set_index, tag), None)

        self.line_insertion_set_access_cnts[(
            set_index, tag)] = self.set_access_cnts[set_index]
        self.line_last_access_type[(set_index, tag)] = access_type

        self.line_access_cnts[(set_index, tag)] = {
            at: 0 for at in ['l', 'r', 'w', 'p', 't']}
        self.line_access_cnts[(set_index, tag)][access_type] = 1

        self.line_hits_since_insertion[(set_index, tag)] = 0
        self.update_set_recency(set_index, tag)

        return True

    def reset(self) -> Tuple[npt.NDArray, bool]:
        self.cache.reset()

        self.total_access_cnt = 0
        self.total_miss_cnt = 0

        self.set_access_cnts = [0 for _ in range(len(self.cache))]
        self.set_access_cnts_at_miss = [0 for _ in range(len(self.cache))]

        self.address_preuses = {}

        self.line_preuses_0 = {}
        self.line_preuses_1 = {}
        self.line_insertion_set_access_cnts = {}
        self.line_last_access_type = {}
        self.line_access_cnts = {}
        self.line_hits_since_insertion = {}
        self.line_recency = {}

        self.curr_trace_idx = self.execute_till_non_compulsory_miss(0)

        if self.curr_trace_idx is None:
            return np.zeros(self.observation_space()), True

        return self.observe(self.curr_trace_idx), False

    def step(self, action: int) -> Tuple[npt.NDArray, float, bool]:
        reward = 0

        next_access_indices = self.belady_replacement_optimal(
            self.curr_trace_idx)

        if next_access_indices is not None:
            reward = 1 if next_access_indices[action] == min(
                next_access_indices) else -1

        success = self.execute_action_on_non_compulsory_miss(
            self.curr_trace_idx, action)

        if not success:
            raise ValueError('Invalid action')

        self.curr_trace_idx = self.execute_till_non_compulsory_miss(
            self.curr_trace_idx + 1)

        if self.curr_trace_idx is None:
            return np.zeros(self.observation_space()), reward, True

        return self.observe(self.curr_trace_idx), reward, False

    def stats(self) -> Dict[str, float]:
        return {
            'hit_rate': 1 - self.total_miss_cnt / self.total_access_cnt,
            'total_access_cnt': self.total_access_cnt,
            'total_miss_cnt': self.total_miss_cnt,
        }
