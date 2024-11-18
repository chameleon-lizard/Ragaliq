# RAGaliq eval 🥐

A simple open-source multilingual RAG benchmark.

## Data description

I've taken a 90-page Skoltech orientation course, which was presented in English and translated it into German, French, Spanish, Russian and Chinese using Gemini Flash (see translation scripts in `src/data_generation/translate.py` and `src/data_generation/translate_questions.py`). To generate questions, I've chunked the data and extracted factoids using Gemini Flash. Using the same model, I've generated questions, answers to which would be the extracted factoids. After that, I've rated the questions using Gemini Flash by three metrics:

- Usefullness - is the question useful for the student of Skoltech
- Standalone - can the question arise without the chunk, from which it was generated
- Groundedness - can the question be answered only using source chunk and in only one way

Code for question generation is available in `src/data_generation`.

After rating and filtering the questions, I've translated the questions and answers into same five languages using Gemini Flash model.

## Scoring

Evaluation is being done using LLM-as-a-judge. Score rubrics are as follows:

```
Score 0: The response is mention that there is nothing found in the context or documents about the question asked.
Score 1: The response is completely incorrect, inaccurate, and/or not factual. If the response is not about the context altogether and does not resemble the reference answer, thhe response should get this score.
Score 2: The response is mostly incorrect, inaccurate, and/or not factual. If the response is incorrect and contradicts the context, the response should get this score.
Score 3: The response is somewhat correct, accurate, and/or factual. If the response is technically useful, but lacks additional information to be called correct, the answer should get this score.
Score 4: The response is mostly correct, accurate, and factual. If the response adds additional information, which is not explicitly asked in the question (e.g. rambling on another topic), but the response itself is mostly correct, the answer should get this score.
Score 5: The response is completely correct, accurate, and factual. Formatting can be different (e.g. capitalization of some letters), but the sense is the same as in the reference.
```

To score the final models, three different scoring metrics are used:

- Mean judge score
- Mean judge score without zeros
- Weighted judge score

Weighted judge score is calculated using the following formula:

```
final_score = (1 / N * Sum(0, N, weighted_score[judge_score[i]]))**2,

weighted_score = {
    5: 1.0,
    4: 0.8,
    3: 0.6,
    2: 0.2,
    1: 0.0,
    0: 0.4,
}
```

Since no answer is better than bad answer, but still worse than usable or incomplete answer, zero is being weighted with higher score than 1 and 2.

To calculate the results, you can use script located in `src/calculate_results.py`. Current results are available [at the results page](https://github.com/chameleon-lizard/Ragaliq/blob/main/results.md).

## Installation

Create a virtual environment and run the installation from `requirements.txt`.

```
python -m venv venv
source ./venv/bin/activate
pip install -r requirements.txt
```

Additionally, you have to provide API token and API link for the llm judge in `.env` file. Example configuration:

```
TOKEN=sk-or-v1-xxx
JUDGE_API_LINK=https://openrouter.ai/api/v1

JUDGE_MODEL=google/gemini-flash-1.5
```

## Run eval

You can run evaluation of a model directly:

```
usage: eval.py [-h] [--knowledge_base KNOWLEDGE_BASE] [--reader_model_id READER_MODEL_ID]
               [--sampling_params SAMPLING_PARAMS] [--embedder_model_id EMBEDDER_MODEL_ID]
               [--reranker_model_id RERANKER_MODEL_ID] [--lang LANG] [--use_decoder_as_embedder]

RAGaliq eval

options:
  -h, --help            show this help message and exit
  --knowledge_base KNOWLEDGE_BASE
                        Path to knowledge_base
  --reader_model_id READER_MODEL_ID
                        Reader model ID
  --sampling_params SAMPLING_PARAMS
                        Path to sampling params
  --embedder_model_id EMBEDDER_MODEL_ID
                        Embedder model ID
  --reranker_model_id RERANKER_MODEL_ID
                        Reranker model ID
  --lang LANG           Language to evaluate. One of ['all', 'en', 'de', 'fr', 'es', 'ru', 'zh']
  --use_decoder_as_embedder
                        Use decoder as embedder
```

Or you can modify evaluation script:

```
./run_eval.sh
```

Check out the script for the examples of how to run `eval.py`. To run your model, you need to add sampling config to `cfg/`. If you want to modify the RAG pipeline, you can consult with `src/rag.py` file and modify `Chatbot` class to suit your needs.
