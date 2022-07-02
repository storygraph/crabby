import itertools
import math
import re
from typing import List, Pattern, Tuple

import torch.utils.data as torch_data


class PairOutOfBoundsError(Exception):
    """
    Thrown whenever a pair out of bounds is requested.
    """


class EntityPairSentenceSet:
    _MARKER_REGEX = r"<e[0-9]*>[^<]*<\/e[0-9]*>"
    _FILTER_MARKER_REGEX = r"<e[0-9]*>([^<]*)<\/e[0-9]*>"

    _sentences: List[str]
    # _cum_pairs_per_sent is cumulative.
    _cum_pairs_per_sent: List[int]
    _marker_regex: Pattern
    _filter_marker_regex: Pattern

    def __init__(self, sentences: List[str]) -> None:
        self._sentences = sentences
        self._marker_regex = re.compile(self._MARKER_REGEX)
        self._filter_marker_regex = re.compile(self._FILTER_MARKER_REGEX)

        self._cum_pairs_per_sent = self._get_pairs_per_sent()

    def __len__(self) -> int:
        if len(self._cum_pairs_per_sent) == 0:
            return 0

        return self._cum_pairs_per_sent[len(self._cum_pairs_per_sent) - 1]

    def __getitem__(self, idx) -> str:
        if not self._exists_pair(idx):
            raise PairOutOfBoundsError(f"Non-existing pair at pos {idx}")

        sent_idx = self._sent_idx_for_pair_at(idx)
        num_sent_pairs = self._num_pairs_for_sent(sent_idx)
        pair_idx = idx - (self._cum_pairs_per_sent[sent_idx] - num_sent_pairs)

        exc_indices = self._pair_ids_for_comb_idx(sent_idx, pair_idx)
        
        return self._filter_markers(sent_idx, exc_indices)
    
    def _get_pairs_per_sent(self) -> List[int]:
        total_pairs = 0
        cum_pairs_per_sent = []
        
        for sent in self._sentences:
            matches = re.findall(self._marker_regex, sent)
            pairs_count = math.comb(len(matches), 2)

            total_pairs += pairs_count
            cum_pairs_per_sent.append(total_pairs)

        return cum_pairs_per_sent

    def _exists_pair(self, idx) -> bool:
        return idx >= 0 and idx < len(self)

    def _sent_idx_for_pair_at(self, idx: int) -> int:
        left = 0
        right = len(self._cum_pairs_per_sent) - 1

        # Done because every item of _cum_pairs_per_sent contains the number of
        # pairs for the given sentence.
        idx += 1

        while left < right:
            mid = left + (right - left) // 2

            if self._cum_pairs_per_sent[mid] == idx:
                return mid
            elif self._cum_pairs_per_sent[mid] < idx:
                left = mid + 1
            else:
                right = mid

        # left can never become greater than right and we have a sparse representation where
        # an idx of a triplet would most probably not be found in an array but is a valid idx.
        return left

    def _num_pairs_for_sent(self, sent_idx: int) -> int:
        if sent_idx == 0:
            return self._cum_pairs_per_sent[0]
        
        return self._cum_pairs_per_sent[sent_idx] - self._cum_pairs_per_sent[sent_idx - 1]

    def _pair_ids_for_comb_idx(self, sent_idx: int, pair_idx: int) -> Tuple[int, int]:
        matches = re.findall(self._MARKER_REGEX, self._sentences[sent_idx])
        nums = [0] * len(matches)
        
        for i in range(len(matches)):
            nums[i] = i+1
    
        combos = list(itertools.combinations(nums, 2))
        return combos[pair_idx]

    def _filter_markers(self, sent_idx: int, exc_indices: Tuple[int, int]) -> str:
        sent = self._sentences[sent_idx]
        matches = re.findall(self._MARKER_REGEX, self._sentences[sent_idx])
        
        matches = list(filter(lambda x: not(x.startswith(f"<e{exc_indices[0]}>") or x.startswith(f"<e{exc_indices[1]}>")), matches))
        
        # This shouldn't have a big timecomplexity.
        # After all we don't compile new regexs all the time.
        for match in matches:
            # we have a guarantee that it is going to be on len 1.
            filtered_content = re.findall(self._filter_marker_regex, match)[0]
            sent = sent.replace(match, filtered_content)
        
        return sent


class SentenceDataset(torch_data.Dataset):
    def __init__(self) -> None:
        super().__init__()

    def __len__(self) -> int:
        return 0

    def __getitem__(self, idx):
        return None
