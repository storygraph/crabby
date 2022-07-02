import unittest

import crabby.rel as crabby_rel


class TestEntityPairSentenceSet(unittest.TestCase):
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
        self._four_ent_sent = "<e1>Oliver</e1> kissed <e2>Sally</e2> for <e3>Christmas eve</e3> in front of father <e4>Heuston</e4>."

    def test_len_of_empty_set(self) -> None:
        sentences = crabby_rel.EntityPairSentenceSet([])
        self.assertEqual(len(sentences), 0)

    def test_len(self) -> None:
        sentences = crabby_rel.EntityPairSentenceSet([self._zero_ent_sent, self._one_ent_sent, self._two_ent_sent])
        self.assertEqual(len(sentences), 1)

    def test_len_sent_with_multiple_entities(self) -> None:
        sentences = crabby_rel.EntityPairSentenceSet([self._four_ent_sent])
        # It should be a combination n=num_entities and k=2 (as we extract pairs)
        self.assertEqual(len(sentences), 6)

    def test_getitem_non_existing_on_empty_set(self) -> None:
        sentences = crabby_rel.EntityPairSentenceSet([])
        
        with self.assertRaises(crabby_rel.PairOutOfBoundsError):
            sentences[0]

    def test_getitem_non_existing(self) -> None:
        sentences = crabby_rel.EntityPairSentenceSet([self._zero_ent_sent, self._one_ent_sent, self._two_ent_sent])
        
        with self.assertRaises(crabby_rel.PairOutOfBoundsError):
            sentences[1]

    def test_getitem(self) -> None:
        sentences = crabby_rel.EntityPairSentenceSet([self._zero_ent_sent, self._one_ent_sent, self._two_ent_sent])

        self.assertEqual(sentences[0], "<e1>John</e1> is a father of <e2>Gordon</e2>.")

    def test_getitem_filters_entity_markers(self) -> None:
        sentences = crabby_rel.EntityPairSentenceSet([self._zero_ent_sent, self._two_ent_sent, self._one_ent_sent ,self._four_ent_sent])
        
        self.assertEqual(len(sentences), 7)
        
        self.assertEqual(sentences[0], "<e1>John</e1> is a father of <e2>Gordon</e2>.")
        
        self.assertEqual(sentences[1], "<e1>Oliver</e1> kissed <e2>Sally</e2> for Christmas eve in front of father Heuston.")
        self.assertEqual(sentences[2], "<e1>Oliver</e1> kissed Sally for <e3>Christmas eve</e3> in front of father Heuston.")
        self.assertEqual(sentences[3], "<e1>Oliver</e1> kissed Sally for Christmas eve in front of father <e4>Heuston</e4>.")
        self.assertEqual(sentences[4], "Oliver kissed <e2>Sally</e2> for <e3>Christmas eve</e3> in front of father Heuston.")
        self.assertEqual(sentences[5], "Oliver kissed <e2>Sally</e2> for Christmas eve in front of father <e4>Heuston</e4>.")
        self.assertEqual(sentences[6], "Oliver kissed Sally for <e3>Christmas eve</e3> in front of father <e4>Heuston</e4>.")

    def test_getitem_filters_reversed_entity_markers(self) -> None:
        sentences = crabby_rel.EntityPairSentenceSet([self._rev_two_ent_sent])
        
        self.assertEqual(len(sentences), 1)
        self.assertEqual(sentences[0], "<e2>John</e2> is a father of <e1>Gordon</e1>.")
