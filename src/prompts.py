EVALUATION_PROMPT = """###Task Description:
An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 0 and 5. You should refer to the score rubric.
3. The output format should look as follows: \"Feedback: {{write a feedback for criteria}} [RESULT] {{an integer number between 0 and 5}}\"
4. Please do not generate any other opening, closing, and explanations. Be sure to include [RESULT] in your output.

###The instruction to evaluate:
{instruction}

###Response to evaluate:
{response}

###Reference Answer (Score 5):
{reference_answer}

###Score Rubrics:
[Is the response correct, accurate, and factual based on the reference answer?]
Score 0: The response is mention that there is nothing found in the context or documents about the question asked.
Score 1: The response is completely incorrect, inaccurate, and/or not factual. If the response is not about the context altogether and does not resemble the reference answer, thhe response should get this score.
Score 2: The response is mostly incorrect, inaccurate, and/or not factual. If the response is incorrect and contradicts the context, the response should get this score.
Score 3: The response is somewhat correct, accurate, and/or factual. If the response is technically useful, but lacks additional information to be called correct, the answer should get this score.
Score 4: The response is mostly correct, accurate, and factual. If the response adds additional information, which is not explicitly asked in the question (e.g. rambling on another topic), but the response itself is mostly correct, the answer should get this score.
Score 5: The response is completely correct, accurate, and factual. Formatting can be different (e.g. capitalization of some letters), but the sense is the same as in the reference.

###Feedback:"""
