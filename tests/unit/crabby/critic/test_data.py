from typing import List
import unittest

import torch
import torch.utils.data as torch_data

import crabby.critic as critic


class TestOntology(unittest.TestCase):
    _adj_list: List[List[critic.Trans]]
    _entities: List[str]
    _rels: List[str]
    
    def setUp(self) -> None:
        self._adj_list = [[critic.Trans(rel=0, tail=1)], [critic.Trans(rel=1, tail=0), critic.Trans(rel=1, tail=1)]]
        self._entities = ["a", "b"]
        self._rels = ["x", "y"]

    def test_exists(self) -> None:
        onto = critic.Ontology(self._adj_list, self._rels, self._entities)

        self.assertTrue(onto.exists(triplet=critic.Triplet(head=0, rel=0, tail=1)))

    def test_not_exists(self) -> None:
        onto = critic.Ontology(self._adj_list, self._rels, self._entities)
        
        self.assertFalse(onto.exists(triplet=critic.Triplet(head=0, rel=1, tail=1)))
        self.assertFalse(onto.exists(triplet=critic.Triplet(head=1, rel=0, tail=1)))

    def test_entities_len(self) -> None:
        onto = critic.Ontology(self._adj_list, self._rels, self._entities)
        
        self.assertEqual(onto.entities_len(), 2)

    def test_triplets_len(self) -> None:
        onto = critic.Ontology(self._adj_list, self._rels, self._entities)
        
        self.assertEqual(onto.triplets_len(), 3)

    def test_get_triplet(self) -> None:
        onto = critic.Ontology(self._adj_list, self._rels, self._entities)
        
        self.assertEqual(onto.get_triplet(0), critic.Triplet(head=0, rel=0, tail=1))
        self.assertEqual(onto.get_triplet(1), critic.Triplet(head=1, rel=1, tail=0))
        self.assertEqual(onto.get_triplet(2), critic.Triplet(head=1, rel=1, tail=1))

    def test_get_non_existing_triplet(self) -> None:
        onto = critic.Ontology(self._adj_list, self._rels, self._entities)
        
        with self.assertRaises(critic.TripletOutOfBoundsError):
            onto.get_triplet(1000)

    def test_add_triplets(self) -> None:
        onto = critic.Ontology(self._adj_list, self._rels, self._entities)
        
        foo_triplet = critic.Triplet(head=0, rel=1, tail=1)
        bar_triplet = critic.Triplet(head=1, rel=0, tail=0)
        onto.add_triplets([foo_triplet, bar_triplet])
        
        self.assertEqual(onto.triplets_len(), 5)

        self.assertEqual(onto.get_triplet(0), critic.Triplet(head=0, rel=0, tail=1))
        self.assertEqual(onto.get_triplet(1), foo_triplet)

        self.assertEqual(onto.get_triplet(2), critic.Triplet(head=1, rel=1, tail=0))
        self.assertEqual(onto.get_triplet(3), critic.Triplet(head=1, rel=1, tail=1))
        self.assertEqual(onto.get_triplet(4), bar_triplet)

    def test_add_triplet_with_non_existing_head(self) -> None:
        onto = critic.Ontology(self._adj_list, self._rels, self._entities)
        
        with self.assertRaises(critic.TripletOutOfBoundsError):
            onto.add_triplets([critic.Triplet(head=10, rel=0, tail=1)])

    def test_add_triplet_with_non_existing_rel(self) -> None:
        onto = critic.Ontology(self._adj_list, self._rels, self._entities)
        
        with self.assertRaises(critic.TripletOutOfBoundsError):
            onto.add_triplets([critic.Triplet(head=0, rel=20, tail=1)])

    def test_add_triplet_with_non_existing_tail(self) -> None:
        onto = critic.Ontology(self._adj_list, self._rels, self._entities)
        
        with self.assertRaises(critic.TripletOutOfBoundsError):
            onto.add_triplets([critic.Triplet(head=0, rel=0, tail=30)])

    def test_validation_of_adj_list(self) -> None:
        self._entities.append("crab")
        
        with self.assertRaises(Exception):
            critic.Ontology(self._adj_list, self._rels, self._entities)


class TestTripletDataset(unittest.TestCase):
    _onto: critic.Ontology
    _adj_list: List[List[critic.Trans]]
    _entities: List[str]
    _rels: List[str]
    
    def setUp(self) -> None:
        self._adj_list = [[critic.Trans(rel=0, tail=1)], [critic.Trans(rel=1, tail=0), critic.Trans(rel=1, tail=1)]]
        self._entities = ["a", "b"]
        self._rels = ["x", "y"]
        self._onto = critic.Ontology(self._adj_list, self._rels, self._entities)

    def test_len(self) -> None:
        dataset = critic.TripletDataset(self._onto)
        
        self.assertEqual(len(dataset), 3)

    def test_getitem(self) -> None:
        dataset = critic.TripletDataset(self._onto)
        
        self._assert_tensor_triplet_equals(dataset[1], expected_triplet=critic.Triplet(head=1, rel=1, tail=0))

    # not exactly a unit test but useful...
    def test_triplet_dataloader(self) -> None:
        loader = torch_data.DataLoader(critic.TripletDataset(self._onto), batch_size=1, shuffle=False)
        
        triplet = torch.squeeze(next(iter(loader)))
        
        self._assert_tensor_triplet_equals(triplet, expected_triplet=critic.Triplet(head=0, rel=0, tail=1))

    def _assert_tensor_triplet_equals(self, triplet: torch.IntTensor, expected_triplet: critic.Triplet) -> None:
        expected_tensor_triplet = torch.tensor([expected_triplet.head, expected_triplet.rel, expected_triplet.tail])
        
        for i in range(3):
            self.assertEqual(triplet[i].item(), expected_tensor_triplet[i])
