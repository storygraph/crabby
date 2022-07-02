import os

import compress_fasttext.models as ft

import crabby.rel as rel


def main():    
    semval_loader = rel.SemvalDatasetLoader()
    pairer = semval_loader.load_dataset()
    
    fasttext = ft.CompressedFastTextKeyedVectors.load(os.getenv("FT_MDL", ""))
    
    training_dataset = rel.SentenceDataset(pairer, fasttext)


if __name__ == "__main__":
    main()
