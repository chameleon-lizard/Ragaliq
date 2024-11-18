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

SYSTEM_PROMPT = {
    "en": "You will be given documents and a question. Your task is to answer the question using these documents. Be factual and only use information from the context to answer the questions. Be concise in your answers, not more than one sentence.",
    "de": "Sie erhalten Unterlagen und eine Frage. Ihre Aufgabe ist es, die Frage anhand dieser Dokumente zu beantworten. Seien Sie sachlich und verwenden Sie nur Informationen aus dem Kontext, um die Fragen zu beantworten. Seien Sie präzise in Ihren Antworten, nicht mehr als einen Satz.",
    "fr": "Vous recevrez des documents et une question. Votre tâche est de répondre à la question en utilisant ces documents. Soyez factuel et n'utilisez que des informations contextuelles pour répondre aux questions. Soyez concis dans vos réponses, pas plus d'une phrase.",
    "es": "Se le entregarán documentos y una pregunta. Su tarea es responder a la pregunta utilizando estos documentos. Sea objetivo y solo use información del contexto para responder las preguntas. Sea conciso en sus respuestas, no más de una oración.",
    "ru": "Вам будут предоставлены документы и вопрос. Ваша задача - ответить на вопрос, используя эти документы. При ответе на вопросы используйте факты и только информацию из контекста. Будьте кратки в своих ответах, не более одного предложения.",
    "zh": "你会得到文件和一个问题。 您的任务是使用这些文档回答问题。 实事求是，只使用上下文中的信息来回答问题。 回答要简洁，不要超过一句话。",
}

GEMMA_RESPONSE_PROMPT = {
    "en": "Okay! Send me the context and the question.",
    "de": "Ok! Senden Sie mir den Kontext und die Frage.",
    "fr": "D'accord! Envoyez-moi le contexte et la question.",
    "es": "¡Bien! Envíame el contexto y la pregunta.",
    "ru": "Хорошо! Пришлите мне контекст и вопрос.",
    "zh": "好吧！ 把上下文和问题发给我。",
}

DOC_SIM_PROMPT = {
    "en": "DOCUMENT SIMILARITY",
    "de": "ÄHNLICHKEIT DES DOKUMENTS",
    "fr": "SIMILARITÉ DES DOCUMENTS",
    "es": "SIMILITUD DE DOCUMENTOS",
    "ru": "СХОДСТВО ДОКУМЕНТОВ",
    "zh": "文档相似性",
}

DOC_TEXT_PROMPT = {
    "en": "DOCUMENT TEXT",
    "de": "DOKUMENTENTEXT",
    "fr": "TEXTE DU DOCUMENT",
    "es": "TEXTO DEL DOCUMENTO",
    "ru": "ТЕКСТ ДОКУМЕНТА",
    "zh": "文档文本",
}

CONTEXT_PROMPT = {
    "en": "CONTEXT",
    "de": "KONTEXT",
    "fr": "CONTEXTE",
    "es": "CONTEXTO",
    "ru": "КОНТЕКСТ",
    "zh": "上下文环境",
}

QUESTION_PROMPT = {
    "en": "QUESTION",
    "de": "FRAGE",
    "fr": "QUESTION",
    "es": "PREGUNTA",
    "ru": "ВОПРОС",
    "zh": "问题",
}
