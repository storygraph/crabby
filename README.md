# Crabby

You think you're a good at storytelling... ehh??? Come and fight me you #$@#@#$#%!*@#$! :crab: :crab: :crab:

`crabby` is a literature critic of some kind. It does so by first recognising the named entities and extracting their relations from a raw text. By doing so it extracts them one by one which enables the creation of a `storygraph` data structure. Once that is done we have an ontology of the named entities and their relations! Isn't this great? Furthermore it employs an easy to train model for multi-relation data called [Transe](https://proceedings.neurips.cc/paper/2013/file/1cecc7a77928ca8133fa24680a88d2f9-Paper.pdf) to create a GNN-like model for link-prediction. Since the embeddings are in a latent vector space then distances between entities could be calculated to obtain a truthfulness score for a relation. This score is then used for criticising a text regarding the story.


## Short dev guide

```bash
# Create a python venv like so.
python3 -m venv .venv

# Install the requirements like so.
(source .venv/bin/activate && pip install -r requirements.txt)

# To test transe run.
make test-transe

# To run unit tests run.
make test-unit
```
