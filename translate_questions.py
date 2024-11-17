import src.utils as utils
import os
import pathlib
import threading
import json
from queue import Queue
import time

import dotenv

dotenv.load_dotenv(".env")


def translate(idx, text, language, q, q_lock, sem):
    judge_model = f"{os.environ.get('JUDGE_MODEL')}"
    judge_api_link = f"{os.environ.get('JUDGE_API_LINK')}"
    token = f"{os.environ.get('TOKEN')}"

    with sem:
        context = utils.send_question(
            prompt=f"Translate the following text into {language}. Do not write anything besides the translation. If the text is in caps, do not translate it. Do not translate links and emails, just return them as is. The text to translate:\n\n{text['context']}",
            model=judge_model,
            api_link=judge_api_link,
            token=token,
            temperature=0.001,
            max_tokens=4096,
        )
        question = utils.send_question(
            prompt=f"Translate the following text into {language}. Do not write anything besides the translation. If the text is in caps, do not translate it. Do not translate links and emails, just return them as is. The text to translate:\n\n{text['question']}",
            model=judge_model,
            api_link=judge_api_link,
            token=token,
            temperature=0.001,
            max_tokens=4096,
        )
        answer = utils.send_question(
            prompt=f"Translate the following text into {language}. Do not write anything besides the translation. If the text is in caps, do not translate it. Do not translate links and emails, just return them as is. The text to translate:\n\n{text['answer']}",
            model=judge_model,
            api_link=judge_api_link,
            token=token,
            temperature=0.001,
            max_tokens=4096,
        )
        print(f"{language} {idx}/71")
    try:
        item = {
            "source_c": text["context"],
            "source_q": text["question"],
            "source_a": text["answer"],
            "context": context,
            "question": question,
            "answer": answer,
        }
        with q_lock:
            q.put(item)
    except Exception:
        with q_lock:
            item = {
                "source_c": text["context"],
                "source_q": text["question"],
                "source_a": text["answer"],
                "context": text["context"],
                "question": text["question"],
                "answer": text["answer"],
            }
            q.put(item)


def language_to_code(language: str) -> str:
    if language == "russian":
        return "ru"
    if language == "german":
        return "de"
    if language == "french":
        return "fr"
    if language == "spanish":
        return "es"
    if language == "chinese":
        return "zh"


def translate_texts(
    data,
    num_threads: int = 50,
):
    sem = threading.Semaphore(num_threads)

    q_lock = threading.Lock()

    threads = []
    q = Queue()
    languages = ["russian", "german", "french", "spanish", "chinese"]
    for language in languages:
        for idx, item in enumerate(data):
            if item == "":
                continue

            thread = threading.Thread(
                target=translate, args=(idx, item, language, q, q_lock, sem)
            )

            thread.start()
            threads.append(thread)
            time.sleep(0.01)

        [_.join() for _ in threads]

        res = []
        while not q.empty():
            res.append(q.get())

        with open(f"data/questions_{language_to_code(language)}.json", "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    data = json.loads(pathlib.Path("data/questions_en.json").read_text())
    translate_texts(data)
