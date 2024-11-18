#!/bin/bash


# Google Gemma 2 2b
python eval.py --reader_model_id google/gemma-2-2b-it --sampling_params cfg/gemma_2_2b.json --embedder_model_id ai-forever/ru-en-RoSBERTa --reranker_model_id BAAI/bge-reranker-v2-m3 --lang en
python eval.py --reader_model_id google/gemma-2-2b-it --sampling_params cfg/gemma_2_2b.json --embedder_model_id BAAI/bge-large-en-v1.5 --reranker_model_id BAAI/bge-reranker-v2-m3 --lang en

python eval.py --reader_model_id google/gemma-2-2b-it --sampling_params cfg/gemma_2_2b.json --embedder_model_id deepvk/USER-bge-m3 --reranker_model_id BAAI/bge-reranker-v2-m3 --lang ru 
python eval.py --reader_model_id google/gemma-2-2b-it --sampling_params cfg/gemma_2_2b.json --embedder_model_id ai-forever/ru-en-RoSBERTa --lang ru

# Vikhr Gemma 2 2b
python eval.py --reader_model_id Vikhrmodels/Vikhr-Gemma-2B-instruct --sampling_params cfg/gemma_2_2b.json --embedder_model_id ai-forever/ru-en-RoSBERTa --reranker_model_id BAAI/bge-reranker-v2-m3 --lang en
python eval.py --reader_model_id Vikhrmodels/Vikhr-Gemma-2B-instruct --sampling_params cfg/gemma_2_2b.json --embedder_model_id BAAI/bge-large-en-v1.5 --reranker_model_id BAAI/bge-reranker-v2-m3 --lang en

python eval.py --reader_model_id Vikhrmodels/Vikhr-Gemma-2B-instruct --sampling_params cfg/gemma_2_2b.json --embedder_model_id deepvk/USER-bge-m3 --reranker_model_id BAAI/bge-reranker-v2-m3 --lang ru 
python eval.py --reader_model_id Vikhrmodels/Vikhr-Gemma-2B-instruct --sampling_params cfg/gemma_2_2b.json --embedder_model_id ai-forever/ru-en-RoSBERTa --lang ru


# Qwen 2.5 7b
python eval.py --reader_model_id Qwen/Qwen2.5-7B-Instruct --sampling_params cfg/qwen_25_7b.json --embedder_model_id ai-forever/ru-en-RoSBERTa --reranker_model_id BAAI/bge-reranker-v2-m3 --lang en
python eval.py --reader_model_id Qwen/Qwen2.5-7B-Instruct --sampling_params cfg/qwen_25_7b.json --embedder_model_id BAAI/bge-large-en-v1.5 --reranker_model_id BAAI/bge-reranker-v2-m3 --lang en

python eval.py --reader_model_id Qwen/Qwen2.5-7B-Instruct --sampling_params cfg/qwen_25_7b.json --embedder_model_id deepvk/USER-bge-m3 --reranker_model_id BAAI/bge-reranker-v2-m3 --lang ru 
python eval.py --reader_model_id Qwen/Qwen2.5-7B-Instruct --sampling_params cfg/qwen_25_7b.json --embedder_model_id ai-forever/ru-en-RoSBERTa --lang ru

# Ruadapt Qwen 2.5 7b
python eval.py --reader_model_id RefalMachine/ruadapt_qwen2.5_7B_ext_u48_instruct --sampling_params cfg/qwen_25_7b.json --embedder_model_id ai-forever/ru-en-RoSBERTa --reranker_model_id BAAI/bge-reranker-v2-m3 --lang en
python eval.py --reader_model_id RefalMachine/ruadapt_qwen2.5_7B_ext_u48_instruct --sampling_params cfg/qwen_25_7b.json --embedder_model_id BAAI/bge-large-en-v1.5 --reranker_model_id BAAI/bge-reranker-v2-m3 --lang en

python eval.py --reader_model_id RefalMachine/ruadapt_qwen2.5_7B_ext_u48_instruct --sampling_params cfg/qwen_25_7b.json --embedder_model_id deepvk/USER-bge-m3 --reranker_model_id BAAI/bge-reranker-v2-m3 --lang ru 
python eval.py --reader_model_id RefalMachine/ruadapt_qwen2.5_7B_ext_u48_instruct --sampling_params cfg/qwen_25_7b.json --embedder_model_id ai-forever/ru-en-RoSBERTa --lang ru


# All langs just for statistics' sake
python eval.py --reader_model_id Qwen/Qwen2.5-7B-Instruct --sampling_params cfg/qwen_25_7b.json --embedder_model_id intfloat/multilingual-e5-large-instruct --reranker_model_id BAAI/bge-reranker-v2-m3 --lang all
python eval.py --reader_model_id RefalMachine/ruadapt_qwen2.5_7B_ext_u48_instruct --sampling_params cfg/qwen_25_7b.json --embedder_model_id intfloat/multilingual-e5-large-instruct --reranker_model_id BAAI/bge-reranker-v2-m3 --lang all
python eval.py --reader_model_id Vikhrmodels/Vikhr-Gemma-2B-instruct --sampling_params cfg/gemma_2_2b.json --embedder_model_id intfloat/multilingual-e5-large-instruct --reranker_model_id BAAI/bge-reranker-v2-m3 --lang all
python eval.py --reader_model_id google/gemma-2-2b-it --sampling_params cfg/gemma_2_2b.json --embedder_model_id intfloat/multilingual-e5-large-instruct --reranker_model_id BAAI/bge-reranker-v2-m3 --lang all
python eval.py --reader_model_id google/gemma-2-9b-it --sampling_params cfg/gemma_2_2b.json --embedder_model_id intfloat/multilingual-e5-large-instruct --reranker_model_id BAAI/bge-reranker-v2-m3 --lang all
python eval.py --reader_model_id CohereForAI/aya-expanse-8b --sampling_params cfg/aya-expanse-8b.json --embedder_model_id intfloat/multilingual-e5-large-instruct --reranker_model_id BAAI/bge-reranker-v2-m3 --lang all

python src/calculate_results.py > results.md
