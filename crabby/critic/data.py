from typing import List, NamedTuple, Tuple

import torch.utils.data as torch_data


class TripletOutOfBoundsError(Exception):
    """
    Thrown whenever a triplet is wrongly indexed.
    """


class Triplet(NamedTuple):
    head: int
    rel: int
    tail: int


# Trans is a synonym for neighbour sometimes... Don't ask me why...
class Trans(NamedTuple):
    rel: int
    tail: int


# Ontology represents the graph as a store of triplets by storing it as an adjacency list.
# This provides some faster searches...
class Ontology:
    # _adj_list should be of len _entities
    _adj_list: List[List[Trans]]
    # _triplet_counts has the number of encountered triplets in _adj_list after a head is encountered.
    # It is cumulative.
    # _triplet_counts has a len of len(_entities) where the last item contains the total number of triplets.
    # It is used for faster indexing (O(log n) because of binary search) and triplet counting (O(1)).
    _triplet_counts: List[int]
    # _relations holds the names of the relations.
    _relations: List[str]
    # _entities holds the names of the entities.
    _entities: List[str]

    def __init__(
        self,
        adj_list: List[List[Trans]], 
        relations: List[str],
        entities: List[str],
    ) -> None:
        self._adj_list = adj_list
        self._relations = relations
        self._entities = entities

        self._validate_adj_list()
        self._triplet_counts = self._count_triplets()

    def exists(self, triplet: Triplet) -> bool:
        if not self._entity_exists(triplet.head):
            return False

        for trans in self._adj_list[triplet.head]:
            if trans.rel == triplet.rel and trans.tail == triplet.tail:
                return True

        return False

    # Adding all triplets at once is faster because on each add we need
    # to update the total triplet counts for correct faster searching.
    def add_triplets(self, triplets: List[Triplet]) -> None:
        for _, triplet in enumerate(triplets):
            if not self._entity_exists(triplet.head):
                raise TripletOutOfBoundsError(f"expected head to be between 0 and {len(self._entities) - 1} but was {triplet.head}")

            if not self._entity_exists(triplet.tail):
                raise TripletOutOfBoundsError(f"expected tail to be between 0 and {len(self._entities) - 1} but was {triplet.tail}")

            if not self._rel_exists(triplet.rel):
                raise TripletOutOfBoundsError(f"expected rel to be between 0 and {len(self._relations) - 1} but was {triplet.rel}")

            self._adj_list[triplet.head].append(Trans(rel=triplet.rel, tail=triplet.tail))

        # Update the length of the graph
        self._triplet_counts = self._count_triplets()

    def triplets_len(self) -> int:
        return self._triplet_counts[len(self._triplet_counts) - 1]

    def get_triplet(self, entity_idx: int) -> Triplet:
        head = self._head_for_triplet_at(entity_idx)

        neigbours = self._adj_list[head]
        trans_idx = entity_idx - (self._triplet_counts[head] - len(neigbours))

        trans = neigbours[trans_idx]
        
        return Triplet(head=entity_idx, rel=trans.rel, tail=trans.tail)

    def _entity_exists(self, entity: int) -> bool:
        return entity >= 0 and entity <= len(self._entities) - 1

    def _rel_exists(self, rel: int) -> bool:
        return rel >= 0 and rel <= len(self._relations) - 1

    def _validate_adj_list(self) -> None:
        if len(self._adj_list) != len(self._entities):
            raise Exception(f"adj list length expected to be {len(self._entities)}, but was {len(self._adj_list)}")

    def _count_triplets(self) -> List[int]:
        total_triplets = 0
        triplet_counts = []

        for head in range(len(self._adj_list)):
            total_triplets += len(self._adj_list[head])
            triplet_counts.append(total_triplets)

        return triplet_counts

    def _head_for_triplet_at(self, idx: int) -> int:
        left = 0
        right = len(self._triplet_counts) - 1
        
        if not self._entity_exists(idx):
            raise TripletOutOfBoundsError(f"Expected triplet idx {idx} to be between 0 and {self._triplet_counts[right] - 1} inclusively.")
        
        # Done because every item of _triplet_counts contains the number of
        # triplets for the given head.
        idx += 1
        
        while left < right:
            mid = left + (right - left) // 2

            if self._triplet_counts[mid] == idx:
                return mid
            elif self._triplet_counts[mid] < idx:
                left = mid + 1
            else:
                right = mid
        
        # left can never become greater than right and we have a sparse representation where
        # an idx of a triplet would most probably not be found in an array but is a valid idx.
        return left


class TripletDataset(torch_data.Dataset):
    pass
