import pathlib
import transformers
import collections

from fast_langdetect import detect

import numpy as np


def get_stats(text, tokenizer):
    text = convert_text(text, tokenizer)
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
        try:
            fallback = detect(token)
        except ValueError:
            fallback = {"lang": "unk", "score": 1.0}
        res = {_["lang"]: _["score"] for _ in [fallback]}

    return sorted(
        res,
        key=lambda x: res[x],
        reverse=True,
    )[0]


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
    reference_answer_languages = [
        (token, classify_token(token, prob_dict)) for token in input_tokens
    ]

    return [_[1] for _ in reference_answer_languages]


def calculate_language_consistency(
    generated_answer_probs,  # List of top-10 logits/probabilities for each token in the generated answer
    reference_answer_tokens,  # List of tokens in the reference answer
    reference_answer_languages,  # List of language labels corresponding to each token in the reference answer
    language_classifier,  # A pre-trained language classifier for token-level language classification (optional)
):
    # Step 1: Calculate the language distribution for the reference answer
    reference_language_dist = collections.Counter(reference_answer_languages)
    total_tokens = len(reference_answer_tokens)
    for lang in reference_language_dist:
        reference_language_dist[
            lang
        ] /= total_tokens  # Normalize to get the distribution

    # Step 2: Initialize accumulators
    total_score = 0.0
    num_tokens = len(generated_answer_probs)

    for i in range(num_tokens):
        # Get the top-10 probabilities and corresponding tokens for this token in the generated answer
        top_10_probs = generated_answer_probs[i]

        # Initialize probability accumulators
        max_correct_prob = 0.0
        max_incorrect_prob = 0.0
        all_probs = []

        for probs in top_10_probs:
            prob, token = probs["prob"], probs["token"]
            if token == "\n":
                continue
            # Use the language classifier to determine the token's language
            token_language = language_classifier(token)

            # Determine if the token is in the correct language
            expected_matching_prob = reference_language_dist.get(token_language, 0)

            # Track the highest probabilities for correct and incorrect languages
            if expected_matching_prob > 0:
                max_correct_prob = max(max_correct_prob, prob)
            else:
                max_incorrect_prob = max(max_incorrect_prob, prob)

            # Store all probabilities for entropy calculation
            all_probs.append(prob)

        # Certainty score (dominance ratio)
        certainty_score = max_correct_prob / (
            max_correct_prob + max_incorrect_prob + 1e-9
        )

        # Stability adjustment (entropy penalty)
        entropy = -sum(p * np.log(p + 1e-9) for p in all_probs)
        max_entropy = -np.log(1 / len(all_probs)) * len(
            all_probs
        )  # Maximum entropy for 10 tokens
        stability_adjustment = 1 - (entropy / max_entropy)

        # Match score
        max_matching_prob = max(
            probs["prob"]
            * reference_language_dist.get(
                language_classifier(probs["token"]),
                0,
            )
            for probs in top_10_probs
            if probs["token"] != "\n"
        )

        # Token score
        token_score = certainty_score * stability_adjustment * max_matching_prob

        # Accumulate the score for this token
        total_score += token_score

    # Step 3: Normalize the final score by the number of tokens in the generated answer
    final_score = total_score / num_tokens

    return final_score


def collect_logprobs(answer):
    collected_logprobs = []
    for token in answer[0].outputs[0].logprobs:
        token_logprobs = []
        for logprob in sorted([token[_] for _ in token], key=lambda x: x.rank):
            token_logprobs.append(
                {"prob": np.exp(logprob.logprob), "token": logprob.decoded_token}
            )
        collected_logprobs.append(token_logprobs)
    return collected_logprobs


if __name__ == "__main__":
    import vllm

    llm = vllm.LLM(
        model="google/gemma-2-2b-it", trust_remote_code=True, max_model_len=4096
    )
    sp = vllm.SamplingParams(max_tokens=20, logprobs=10)
    messages = [
        {
            "role": "user",
            "content": "Какой email у отдела по работе со студентами Сколтеха?",
        }
    ]

    tokenizer = transformers.AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
    prob_dict = get_prob_dict(tokenizer, "data")

    # Reference answer tokens and languages
    reference_answer = "К сожалению, я не имею доступа к реальной информации, включая контакты отделов Сколтеха."
    reference_answer_tokens = convert_text(reference_answer, tokenizer)

    reference_answer_languages = classify_text_by_tokens(
        reference_answer,
        tokenizer,
        prob_dict,
    )

    res = llm.chat(messages, sp)

    generated_answer_probs = collect_logprobs(res)

    score = calculate_language_consistency(
        generated_answer_probs,
        reference_answer_tokens,
        reference_answer_languages,
        language_classifier=lambda _: classify_token(_, prob_dict),
    )
    print(f"Language Consistency Score: {score}")
    print(res[0].outputs[0].text)
