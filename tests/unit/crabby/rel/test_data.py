from typing import List
import unittest

import torch

import crabby.rel as crabby_rel
from crabby.rel.data import LabelError


class TestSentencePairer(unittest.TestCase):
    _zero_ent_sent: str
    _one_ent_sent: str
    _two_ent_sent: str
    _rev_two_ent_sent: str
    _four_ent_sent: str
    
    def setUp(self) -> None:
        self._zero_ent_sent = "I have no entities."
        self._one_ent_sent = "<e1>Mike</e1> went outside."
        self._two_ent_sent = "<e1>John</e1> is a father of <e2>Gordon</e2>."
        self._rev_two_ent_sent = "<e2>John</e2> is a father of <e1>Gordon</e1>."
        self._three_ent_sent = "<e1>Minnie</e1> loves <e2>Mickey</e2> but dislikes <e3>Alberto</e3>!"
        self._four_ent_sent = "<e1>Oliver</e1> kissed <e2>Sally</e2> for <e3>Christmas eve</e3> in front of father <e4>Heuston</e4>."

    def test_len_of_empty_set(self) -> None:
        sentences = crabby_rel.SentencePairer([])
        self.assertEqual(len(sentences), 0)

    def test_len(self) -> None:
        sentences = crabby_rel.SentencePairer([self._zero_ent_sent, self._one_ent_sent, self._two_ent_sent])
        self.assertEqual(len(sentences), 1)

    def test_len_sent_with_multiple_entities(self) -> None:
        sentences = crabby_rel.SentencePairer([self._four_ent_sent])
        # It should be a combination n=num_entities and k=2 (as we extract pairs)
        self.assertEqual(len(sentences), 6)

    def test_getitem_non_existing_on_empty_set(self) -> None:
        sentences = crabby_rel.SentencePairer([])
        
        with self.assertRaises(crabby_rel.PairOutOfBoundsError):
            sentences[0]

    def test_getitem_non_existing(self) -> None:
        sentences = crabby_rel.SentencePairer([self._zero_ent_sent, self._one_ent_sent, self._two_ent_sent])
        
        with self.assertRaises(crabby_rel.PairOutOfBoundsError):
            sentences[1]

    def test_getitem(self) -> None:
        sentences = crabby_rel.SentencePairer([self._zero_ent_sent, self._one_ent_sent, self._two_ent_sent])

        self.assertEqual(sentences[0], "<e1>John</e1> is a father of <e2>Gordon</e2>.")

    def test_getitem_filters_entity_markers(self) -> None:
        sentences = crabby_rel.SentencePairer([self._zero_ent_sent, self._two_ent_sent, self._one_ent_sent ,self._four_ent_sent])
        
        self.assertEqual(len(sentences), 7)
        
        self.assertEqual(sentences[0], "<e1>John</e1> is a father of <e2>Gordon</e2>.")
        
        self.assertEqual(sentences[1], "<e1>Oliver</e1> kissed <e2>Sally</e2> for Christmas eve in front of father Heuston.")
        self.assertEqual(sentences[2], "<e1>Oliver</e1> kissed Sally for <e2>Christmas eve</e2> in front of father Heuston.")
        self.assertEqual(sentences[3], "<e1>Oliver</e1> kissed Sally for Christmas eve in front of father <e2>Heuston</e2>.")
        self.assertEqual(sentences[4], "Oliver kissed <e1>Sally</e1> for <e2>Christmas eve</e2> in front of father Heuston.")
        self.assertEqual(sentences[5], "Oliver kissed <e1>Sally</e1> for Christmas eve in front of father <e2>Heuston</e2>.")
        self.assertEqual(sentences[6], "Oliver kissed Sally for <e1>Christmas eve</e1> in front of father <e2>Heuston</e2>.")

    def test_getitem_filters_reversed_entity_markers(self) -> None:
        sentences = crabby_rel.SentencePairer([self._rev_two_ent_sent])
        
        self.assertEqual(len(sentences), 1)
        self.assertEqual(sentences[0], "<e2>John</e2> is a father of <e1>Gordon</e1>.")

    def test_labels_count_validation(self) -> None:
        with self.assertRaises(LabelError):
            crabby_rel.SentencePairer([self._rev_two_ent_sent], labels=[], relations=[])
            
    def test_relations_missing_with_present_labels(self) -> None:
        with self.assertRaises(LabelError):
            crabby_rel.SentencePairer([], labels=[])

    def test_labels_missing_with_present_relations(self) -> None:
        with self.assertRaises(LabelError):
            crabby_rel.SentencePairer([], relations=[])

    def test_undefined_labels(self) -> None:
        with self.assertRaises(LabelError):
            crabby_rel.SentencePairer([self._two_ent_sent], labels=["love"], relations=[])

    def test_getitem_with_label(self) -> None:
        sentences = crabby_rel.SentencePairer(
            sentences=[self._three_ent_sent],
            labels=["love", "dislike", "none"],
            relations=["love", "dislike", "none"],
        )
        
        self.assertEqual(sentences[0], ("<e1>Minnie</e1> loves <e2>Mickey</e2> but dislikes Alberto!", "love"))
        self.assertEqual(sentences[1], ("<e1>Minnie</e1> loves Mickey but dislikes <e2>Alberto</e2>!", "dislike"))
        self.assertEqual(sentences[2], ("Minnie loves <e1>Mickey</e1> but dislikes <e2>Alberto</e2>!", "none"))


class TestSentenceDataset(unittest.TestCase):
    _pairer: crabby_rel.SentencePairer
    
    def setUp(self) -> None:
        three_ent_sent = "<e1>Minnie</e1> loves <e2>Mickey</e2> but dislikes <e3>Alberto</e3>!"
        
        self._pairer = crabby_rel.SentencePairer(
            [three_ent_sent], 
            labels=["love", "dislike", "none"],
            relations=["love", "dislike", "none"],
        )

    def test_len(self) -> None:
        dataset = crabby_rel.SentenceDataset(self._pairer)
        self.assertEqual(len(dataset), 3)

    def test_getitem(self) -> None:
        dataset = crabby_rel.SentenceDataset(self._pairer)
        
        _, label = dataset[0]
        self._assert_tensor_equals(label, [1.0, 0.0, 0.0])
        
        _, label = dataset[1]
        self._assert_tensor_equals(label, [0.0, 1.0, 0.0])

        _, label = dataset[2]
        self._assert_tensor_equals(label, [0.0, 0.0, 1.0])

    def _assert_tensor_equals(self, t: torch.FloatTensor, expected: List[float]) -> None:
        for i, val in enumerate(t):
            self.assertAlmostEqual(val, expected[i])
