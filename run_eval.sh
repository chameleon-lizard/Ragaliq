#!/bin/bash

python eval.py --reader_model_id google/gemma-2-2b-it --sampling_params cfg/gemma_2_2b.json --embedder_model_id intfloat/multilingual-e5-large-instruct --reranker_model_id BAAI/bge-reranker-v2-m3 --lang en
python eval.py --reader_model_id google/gemma-2-2b-it --sampling_params cfg/gemma_2_2b.json --embedder_model_id intfloat/multilingual-e5-large-instruct --reranker_model_id BAAI/bge-reranker-v2-m3 --lang ru
python src/calculate_results.py > results.md
