#!/bin/bash

python eval.py --reader_model_id Qwen/Qwen2.5-7B-Instruct --sampling_params cfg/qwen_25_7b.json
python eval.py --reader_model_id Qwen/Qwen2.5-7B-Instruct --sampling_params cfg/qwen_25_7b.json --embedder_model_id BAAI/bge-large-en-v1.5 --reranker_model_id BAAI/bge-reranker-v2-m3

python eval.py --reader_model_id unsloth/Meta-Llama-3.1-8B-Instruct --sampling_params cfg/l318b.json
python eval.py --reader_model_id unsloth/Meta-Llama-3.1-8B-Instruct --sampling_params cfg/l318b.json --embedder_model_id BAAI/bge-large-en-v1.5 --reranker_model_id BAAI/bge-reranker-v2-m3

python eval.py --reader_model_id CohereForAI/aya-expanse-8b --sampling_params cfg/aya_expanse_8b.json
python eval.py --reader_model_id CohereForAI/aya-expanse-8b --sampling_params cfg/aya_expanse_8b.json --embedder_model_id BAAI/bge-large-en-v1.5 --reranker_model_id BAAI/bge-reranker-v2-m3

python eval.py --reader_model_id mistralai/Mistral-7B-Instruct-v0.3 --sampling_params cfg/mistral_7b.json
python eval.py --reader_model_id mistralai/Mistral-7B-Instruct-v0.3 --sampling_params cfg/mistral_7b.json --embedder_model_id BAAI/bge-large-en-v1.5 --reranker_model_id BAAI/bge-reranker-v2-m3

python eval.py --reader_model_id unsloth/Llama-3.2-3B-Instruct --sampling_params cfg/l323b.json
python eval.py --reader_model_id unsloth/Llama-3.2-3B-Instruct --sampling_params cfg/l323b.json --embedder_model_id BAAI/bge-large-en-v1.5 --reranker_model_id BAAI/bge-reranker-v2-m3
