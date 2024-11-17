#!/bin/bash

python eval.py --reader_model_id Qwen/Qwen2.5-7B-Instruct --sampling_params cfg/qwen_25_7b.json
python eval.py --reader_model_id Qwen/Qwen2.5-7B-Instruct --sampling_params cfg/qwen_25_7b.json --embedder_model_id BAAI/bge-large-en-v1.5 --reranker_model_id BAAI/bge-reranker-v2-m3
python eval.py --reader_model_id Qwen/Qwen2.5-7B-Instruct --sampling_params cfg/qwen_25_7b.json --embedder_model_id intfloat/multilingual-e5-large-instruct --reranker_model_id BAAI/bge-reranker-v2-m3

python eval.py --reader_model_id CohereForAI/aya-expanse-8b --sampling_params cfg/aya_expanse_8b.json
python eval.py --reader_model_id CohereForAI/aya-expanse-8b --sampling_params cfg/aya_expanse_8b.json --embedder_model_id BAAI/bge-large-en-v1.5 --reranker_model_id BAAI/bge-reranker-v2-m3
python eval.py --reader_model_id CohereForAI/aya-expanse-8b --sampling_params cfg/aya_expanse_8b.json --embedder_model_id intfloat/multilingual-e5-large-instruct --reranker_model_id BAAI/bge-reranker-v2-m3

python eval.py --reader_model_id unsloth/gemma-2-2b --sampling_params cfg/l323b.json
python eval.py --reader_model_id unsloth/gemma-2-2b --sampling_params cfg/l323b.json --embedder_model_id BAAI/bge-large-en-v1.5 --reranker_model_id BAAI/bge-reranker-v2-m3
python eval.py --reader_model_id unsloth/gemma-2-2b --sampling_params cfg/l323b.json --embedder_model_id intfloat/multilingual-e5-large-instruct --reranker_model_id BAAI/bge-reranker-v2-m3

python eval.py --reader_model_id RefalMachine/ruadapt_qwen2.5_7B_ext_u48_instruct --sampling_params cfg/qwen_25_7b.json
python eval.py --reader_model_id RefalMachine/ruadapt_qwen2.5_7B_ext_u48_instruct --sampling_params cfg/qwen_25_7b.json --embedder_model_id BAAI/bge-large-en-v1.5 --reranker_model_id BAAI/bge-reranker-v2-m3
python eval.py --reader_model_id RefalMachine/ruadapt_qwen2.5_7B_ext_u48_instruct --sampling_params cfg/qwen_25_7b.json --embedder_model_id intfloat/multilingual-e5-large-instruct --reranker_model_id BAAI/bge-reranker-v2-m3

python eval.py --reader_model_id AnatoliiPotapov/T-lite-instruct-0.1 --sampling_params cfg/l318b.json
python eval.py --reader_model_id AnatoliiPotapov/T-lite-instruct-0.1 --sampling_params cfg/l318b.json --embedder_model_id BAAI/bge-large-en-v1.5 --reranker_model_id BAAI/bge-reranker-v2-m3
python eval.py --reader_model_id AnatoliiPotapov/T-lite-instruct-0.1 --sampling_params cfg/l318b.json --embedder_model_id intfloat/multilingual-e5-large-instruct --reranker_model_id BAAI/bge-reranker-v2-m3


