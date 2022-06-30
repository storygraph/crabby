import crabby.critic as critic
from crabby.critic.data import Triplet


def main():
    adj_list = [[critic.Trans(rel=0, tail=1)], [critic.Trans(rel=1, tail=0), critic.Trans(rel=1, tail=1)]]
    entities = ["a", "b"]
    rels = ["x", "y"]

    onto = critic.Ontology(adj_list, rels, entities)
    print(onto.triplets_len())
    print(onto.entities_len())
    print(onto.get_triplet(2))
    print(onto.exists(critic.Triplet(rel=0, tail=1, head=1)))
    print(onto.add_triplets(triplets=[Triplet(head=0, rel=1, tail=1)]))
    print(onto.triplets_len(), onto.get_triplet(1))


if __name__ == "__main__":
    main()
