import src.utils as utils
import os
import pathlib
import threading
import json
from queue import Queue
import time
import requests


import dotenv

dotenv.load_dotenv(".env")


def translate(idx, text, language, q, q_lock, sem):
    judge_model = f"{os.environ.get('JUDGE_MODEL')}"
    judge_api_link = f"{os.environ.get('JUDGE_API_LINK')}"
    token = f"{os.environ.get('TOKEN')}"

    with sem:
        if text == "":
            translation = ""
        else:
            translation = utils.send_question(
                prompt=f"Translate the following text into {language}. Do not write anything besides the translation. If the text is in caps, do not translate it. Do not translate links and emails, just return them as is. If the input text is a list, return the list without breaking the enumeration and bullet points. The text to translate:\n\n{text}",
                model=judge_model,
                api_link=judge_api_link,
                token=token,
                temperature=0.001,
                max_tokens=4096,
            )

            print(f"Resp {language}: {idx}/2304")
    try:
        item = {"idx": idx, "source": text, "translation": translation}
        with q_lock:
            q.put(item)
    except Exception:
        return


def is_list_item(item):
    def is_digit(x, y):
        try:
            return x[: y - 1].isdigit() and (x[y] == ")" or x[y] == ".")
        except IndexError:
            return False

    return (
        item.startswith("-")
        or (item.startswith("*") and not item.endswith("*"))
        or is_digit(item, 1)
        or is_digit(item, 2)
        or is_digit(item, 3)
        or is_digit(item, 4)
    )


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
    list_buffer = ""
    list_flag = False
    languages = ["russian", "german", "french", "spanish", "chinese"]
    for language in languages:
        for idx, item in enumerate(data):
            item = item.strip()
            if item == "":
                continue

            if is_list_item(item) and list_flag:
                list_buffer += item + "\n"
                continue

            if is_list_item(item) and not list_flag:
                list_buffer = data[idx - 1] + item + "\n"
                list_flag = True
                continue

            if not is_list_item(item) and list_flag:
                list_buffer += item + "\n"
                list_flag = False
                item = list_buffer
                list_buffer = ""

            try:
                if is_list_item(data[idx + 1]) and not list_flag:
                    continue
            except IndexError:
                pass

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

        pathlib.Path(f"data/orientation_{language_to_code(language)}.md").write_text(
            "\n".join([_["translation"] for _ in sorted(res, key=lambda x: x["idx"])])
        )

        with open(f"data/orientation_{language_to_code(language)}.json", "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    data = pathlib.Path("data/orientation_en.md").read_text().splitlines()
    translate_texts(data)
