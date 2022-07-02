from rdflib import Graph

import crabby.critic as critic

import torch
import torch.utils.data as torch_data


def main():
    # adj_list = [[critic.Trans(rel=0, tail=1)], [critic.Trans(rel=1, tail=0), critic.Trans(rel=1, tail=1)]]
    # entities = ["a", "b"]
    # rels = ["x", "y"]

    # onto = critic.Ontology(adj_list, rels, entities)
    # print(onto.triplets_len())
    # print(onto.entities_len())
    # print(onto.get_triplet(2))
    # print(onto.exists(critic.Triplet(rel=0, tail=1, head=1)))
    # print(onto.add_triplets(triplets=[Triplet(head=0, rel=1, tail=1)]))
    # print(onto.triplets_len(), onto.get_triplet(1))
    
    g = Graph()
    g.parse('https://dbpedia.org/resource/High_Speed_1')
    
    entities = set()
    relations = set()

    for s, p, o in g:
        entities.add(s)
        entities.add(o)
        
        relations.add(p)

    entities = list(entities)
    relations = list(relations)
    
    ent_indices = dict()
    rel_indices = dict()
    
    for i, entity in enumerate(entities):
        ent_indices[entity] = i

    for i, rel in enumerate(relations):
        rel_indices[rel] = i

    adj_list = [None] * len(entities)
    
    for i in range(len(entities)):
        adj_list[i] = []

    for s, p, o in g:
        adj_list[ent_indices[s]].append(critic.Trans(rel=rel_indices[p], tail=ent_indices[o]))

    onto = critic.Ontology(adj_list, relations, entities)
    dataset = critic.TripletDataset(onto)
    loader = torch_data.DataLoader(dataset, shuffle=True, batch_size=64)

    model = critic.TranseModel(onto, k=20)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    trainer = critic.Trainer(loader, onto, optimizer, model, margin=1)
    calc = critic.Calculator(dataset, onto)
    
    metrics_bundle = calc.calculate(model)
    print(f"Hits at 10 ---> {metrics_bundle.hits_at_10}")
    print(f"Mean rank ---> {metrics_bundle.mean_rank}")
    
    for _ in range(100):
        trainer.train_one_epoch()
    
    
    metrics_bundle = calc.calculate(model)
    
    print(f"Hits at 10 ---> {metrics_bundle.hits_at_10}")
    print(f"Mean rank ---> {metrics_bundle.mean_rank}")


if __name__ == "__main__":
    main()
