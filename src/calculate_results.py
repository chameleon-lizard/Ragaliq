import pathlib
import pandas as pd


def calculate_score(grades):
    # Define the weights for each grade
    weights = {
        5: 1.0,  # Best grade
        4: 0.8,  # Good
        3: 0.6,  # Marginally acceptable
        0: 0.4,  # No answer, better than 1 or 2
        2: 0.2,  # Bad
        1: 0.0,  # Worst
    }

    # Map grades to their respective weights
    weighted_scores = [weights[grade] for grade in grades]

    # Calculate the average weighted score
    average_weighted_score = sum(weighted_scores) / len(weighted_scores)

    # Square the result to penalize low scores more heavily
    final_score = average_weighted_score**2

    return final_score


if __name__ == "__main__":
    res = []
    for path in pathlib.Path("logs/eval/").iterdir():
        if "res" not in str(path):
            continue

        df = pd.read_json(path)
        lang = str(path).split("_")[3]
        score = calculate_score(df.score.to_list())
        mean_score = df.score[df.score != 0].mean()
        median_score = df.score[df.score != 0].median()
        percentage = df.score[df.score != 0].mean() / 5 * 100
        percentage_with_zero = df.score.mean() / 5 * 100
        fives = sum(df.score == 5)
        fours = sum(df.score == 4)
        threes = sum(df.score == 3)
        twos = sum(df.score == 2)
        ones = sum(df.score == 1)
        zeros = sum(df.score == 0)
        model_info = " ".join(str(path).split("_")[4:-1])
        model_name, embedder_name, reranker_name = model_info.split()
        reranker_name = reranker_name.split(".json")[0]
        generation_time = (
            pathlib.Path(
                f"logs/res/results_questions_{lang}_{model_name}_{embedder_name}_{reranker_name}_{lang}.txt"
            )
            .read_text()
            .splitlines()[-2]
        )

        res.append(
            (
                model_name,
                embedder_name,
                reranker_name,
                lang,
                score,
                mean_score,
                median_score,
                percentage,
                percentage_with_zero,
                fives,
                fours,
                threes,
                twos,
                ones,
                zeros,
                generation_time,
            )
        )

    df_res = pd.DataFrame(
        res,
        columns=(
            "model_name",
            "embedder_name",
            "reranker_name",
            "lang",
            "score",
            "mean_grade",
            "median_grade",
            "percentage",
            "percentage_with_zero",
            "5",
            "4",
            "3",
            "2",
            "1",
            "0",
            "generation_time",
        ),
    )

    for lang in df_res.lang.unique():
        print("\n## " + lang + "\n")
        df_subres = df_res[df_res.lang == lang]

        print(df_subres.sort_values("score", ascending=False).to_markdown(index=False))
        print()
        print(
            df_subres.sort_values(
                ["model_name", "score"], ascending=[False, False]
            ).to_markdown(index=False)
        )
