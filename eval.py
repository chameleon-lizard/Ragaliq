import argparse
import pathlib
import json

from tqdm import tqdm

import pandas as pd

import os
import dotenv
import threading
import torch
from queue import Queue

import src.prompts as prompts
import src.utils as utils

import time

dotenv.load_dotenv(".env")


def judge(item, q, q_lock, sem):
    judge_model = f"{os.environ.get('JUDGE_MODEL')}"
    judge_api_link = f"{os.environ.get('JUDGE_API_LINK')}"
    token = f"{os.environ.get('TOKEN')}"

    with sem:
        eval = utils.send_question(
            prompt=prompts.EVALUATION_PROMPT.format(
                instruction=item["question"],
                response=item["generated_answer"],
                reference_answer=item["true_answer"],
            ),
            model=judge_model,
            api_link=judge_api_link,
            token=token,
            temperature=0.001,
            max_tokens=512,
        )

    try:
        feedback, score = [i.strip() for i in eval.split("[RESULT]")]
        print(f"Score: {score}\nFeedback: {feedback}")
        item["feedback"] = feedback
        item["score"] = int(score)

        with q_lock:
            q.put(item)
    except Exception:
        return


def generate_answers(
    data: dict,
    c,
    save_filename: str | None = None,
) -> list[dict[str, str]]:
    outputs = []
    for example in tqdm(data):
        question = example["question"]
        if question in [output["question"] for output in outputs]:
            continue

        answer, context = c.send_question(question)

        result = {
            "question": question,
            "true_answer": example["answer"],
            "source_doc": example["context"],
            "context": context,
            "generated_answer": answer,
        }

        outputs.append(result)

    if save_filename is not None:
        with open(f"eval/eval_ans_{save_filename}.json", "w") as f:
            json.dump(outputs, f, indent=2)

    return outputs


def judge_answers(
    flattened_data: list[dict[str, str]],
    num_threads: int = 1,
    save_filename: str | None = None,
) -> list[dict[str, str]]:
    sem = threading.Semaphore(num_threads)

    q_lock = threading.Lock()

    threads = []
    q = Queue()
    for item in tqdm(flattened_data):
        thread = threading.Thread(target=judge, args=(item, q, q_lock, sem))
        thread.start()
        threads.append(thread)

    [_.join() for _ in threads]

    res = []
    while not q.empty():
        res.append(q.get())

    if save_filename is not None:
        with open(f"eval/eval_res_{save_filename}.json", "w") as f:
            json.dump(res, f, indent=2)

    return res


def parse_args():
    parser = argparse.ArgumentParser(description="My Program")
    parser.add_argument(
        "--knowledge_base",
        type=str,
        help="Path to knowledge_base",
        default="data/orientation.md",
    )
    parser.add_argument(
        "--reader_model_id",
        type=str,
        default="google/gemma-2-2b-it",
        help="Reader model ID",
    )
    parser.add_argument(
        "--sampling_params",
        type=str,
        help="Path to sampling params",
        default="cfg/gemma_2_2b.json",
    )

    parser.add_argument(
        "--embedder_model_id",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedder model ID",
    )
    parser.add_argument(
        "--reranker_model_id",
        type=str,
        default="BAAI/bge-reranker-base",
        help="Reranker model ID",
    )
    parser.add_argument(
        "--use_decoder_as_embedder",
        action="store_true",
        default=False,
        help="Use decoder as embedder",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    languages = ["en", "ru", "de", "fr", "es", "zh"]

    args = parse_args()
    for language in languages:
        eval_files = [f"data/questions_{language}.json"]
        text = pathlib.Path(f"data/orientation_{language}.md").read_text()

        if args.embedder_model_id == "wordllama/wordllama":
            from cpurag import Chatbot
        else:
            from main import Chatbot

        c = Chatbot(
            knowledge_base=text,
            reader_model_id=args.reader_model_id,
            sampling_params=args.sampling_params,
            embedder_model_id=args.embedder_model_id,
            reranker_model_id=args.reranker_model_id,
            use_decoder_as_embedder=args.use_decoder_as_embedder,
        )

        evals = []
        times = []
        for eval_file in eval_files:
            data = json.loads(pathlib.Path(eval_file).read_text())
            save_filename = (
                eval_file.split("/")[-1].split(".")[0]
                + "_"
                + args.reader_model_id.split("/")[-1]
                + "_"
                + args.embedder_model_id.split("/")[-1]
                + "_"
                + args.reranker_model_id.split("/")[-1]
                + "_"
                + language
            )

            start_time = time.time()

            flattened_data = generate_answers(
                data=data,
                c=c,
                save_filename=save_filename,
            )

            generation_time = str(time.time() - start_time)
            times.append(generation_time)

            evals.append(
                judge_answers(
                    flattened_data=flattened_data,
                    num_threads=50,
                    save_filename=save_filename,
                )
            )

        for eval, eval_file, t in zip(evals, eval_files, times):
            df = pd.DataFrame(eval)

            res_str = (
                f"Eval file: {eval_file}\n"
                + f"Judge model: {os.environ.get('JUDGE_MODEL')}\n"
                + str(c)
                + str(df.score.value_counts().sort_index(ascending=False))
                + "\n\n"
                + "Mean score: \n"
                + str(df.score[df.score != 0].mean())
                + "\nMedian score: \n"
                + str(df.score[df.score != 0].median())
                + "\nPercentage: \n"
                + str(df.score[df.score != 0].mean() / 5 * 100)
                + "\nPercentage with zero: \n"
                + str(df.score.mean() / 5 * 100)
                + "\nGeneration time: \n"
                + str(t)
                + "\n\n"
            )

            print(res_str)

            save_filename = (
                eval_file.split("/")[-1].split(".")[0]
                + "_"
                + args.reader_model_id.split("/")[-1]
                + "_"
                + args.embedder_model_id.split("/")[-1]
                + "_"
                + args.reranker_model_id.split("/")[-1]
                + "_"
                + language
            )

            pathlib.Path(f"res/results_{save_filename}.txt").write_text(res_str)

        del c
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
