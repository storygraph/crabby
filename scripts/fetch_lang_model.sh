#!/bin/bash


mkdir -p data

pushd data
  wget "https://github.com/avidale/compress-fasttext/releases/download/gensim-4-draft/ft_cc.en.300_freqprune_100K_20K_pq_100.bin"
popd
