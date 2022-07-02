import os

import compress_fasttext.models as ft
import torch.utils.data as torch_data
import torch

import crabby.rel as rel


def main():    
    semval_loader = rel.SemvalDatasetLoader()
    training_pairer, test_pairer = semval_loader.load_dataset()
    
    fasttext = ft.CompressedFastTextKeyedVectors.load(os.getenv("FT_MDL", ""))
    
    training_dataset = rel.SentenceDataset(training_pairer, fasttext)
    # they need to be with size 1 because sentences have different lengths => slow!
    loader = torch_data.DataLoader(training_dataset, batch_size=1, shuffle=True)
    
    data_dir = os.getenv("DATA_DIR", "")
    relex_model_path = os.path.join(data_dir, "relex.pt")
    
    if os.path.exists(relex_model_path):
        with open(relex_model_path, 'br') as stream:
            model = torch.load(stream)
        
        print("loaded model from file")
    else:    
        model = rel.RelexModel(d=300, m=250, r=training_pairer.rel_count())
        print("created new model")

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    trainer = rel.RelexTrainer(loader, optimizer, model)

    for _ in range(40):
        trainer.train_one_epoch()

    with open(relex_model_path, 'bw') as stream:
        torch.save(model, stream)

    test_dataset = rel.SentenceDataset(test_pairer, fasttext)
    test_loader = torch_data.DataLoader(test_dataset, batch_size=1, shuffle=True)
    
    num_success = 0
    total = 0
    
    for sentence, label in test_loader:
        out = model(sentence)

        actual_rel_idx = rel.classify(torch.squeeze(out))
        expected_rel_idx = rel.classify(torch.squeeze(label))
        
        if actual_rel_idx == expected_rel_idx:
            num_success += 1
        
        total += 1
    
    print(f"Accuracy ---> {num_success / total}")


if __name__ == "__main__":
    main()
