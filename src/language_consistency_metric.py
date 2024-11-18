import pathlib
import transformers
import collections

from fast_langdetect import detect


def get_stats(text, tokenizer):
    text = [
        tokenizer.decode(_, skip_special_tokens=True)
        for _ in tokenizer(text)["input_ids"]
    ]
    counts = collections.Counter(text)
    return dict(counts), len(text)


def calculate_normalized_probabilities(data_dict):
    # First, calculate per-language probabilities
    initial_probs = {}
    for lang, (lang_tokens, token_counts) in data_dict.items():
        initial_probs[lang] = {
            token: count / lang_tokens for token, count in token_counts.items()
        }

    # Get all unique tokens
    all_tokens = set()
    for _, token_counts in data_dict.values():
        all_tokens.update(token_counts.keys())

    # Create normalized probabilities
    final_probs = {lang: {} for lang in data_dict.keys()}

    # For each token, normalize its probabilities across languages
    for token in all_tokens:
        # Get probabilities for this token across all languages
        token_probs = {
            lang: initial_probs[lang].get(token, 0) for lang in data_dict.keys()
        }

        # Calculate sum for normalization
        prob_sum = sum(token_probs.values())

        # Normalize probabilities
        if prob_sum > 0:  # Avoid division by zero
            for lang in data_dict.keys():
                final_probs[lang][token] = token_probs[lang] / prob_sum

    return final_probs


def get_prob_dict(tokenizer, data_path):
    res = dict()
    for path in pathlib.Path(data_path).iterdir():
        if not str(path).startswith(f"{data_path}/orientation") or not str(
            path
        ).endswith(".md"):
            continue

        lang = str(path).split(".")[0].split("_")[-1]
        text = path.read_text()

        counts, len = get_stats(text, tokenizer)

        res[lang] = (len, counts)

    return calculate_normalized_probabilities(res)


def classify_token(token, prob_dict):
    probs = dict()
    for lang in prob_dict:
        if token in prob_dict[lang]:
            prob = prob_dict[lang][token]
            probs[lang] = prob

    res = {k: v for k, v in probs.items() if v != 0.0}

    if res == {}:
        fallback = detect(token)
        res = {_["lang"]: _["score"] for _ in [fallback]}

    return res


def convert_text(text, tokenizer):
    res = []
    for input_id in tokenizer(text)["input_ids"]:
        decoded = tokenizer.decode(input_id, skip_special_tokens=True)
        if decoded == "":
            continue
        else:
            res.append(decoded)

    return res


def classify_text_by_tokens(text, tokenizer, prob_dict):
    input_tokens = convert_text(text, tokenizer)

    return [(token, classify_token(token, prob_dict)) for token in input_tokens]


if __name__ == "__main__":
    tokenizer = transformers.AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
    prob_dict = get_prob_dict(tokenizer, "data")

    print(
        classify_text_by_tokens(
            "На каком языке these orientation Skoltech синхрафазатрон",
            tokenizer,
            prob_dict,
        )
    )
