import openai


def send_question(
    prompt: str,
    model: str,
    api_link: str,
    token: str,
    temperature: float,
    max_tokens: int,
):
    client = openai.OpenAI(
        api_key=token,
        base_url=api_link,
    )

    messages = []
    messages.append({"role": "user", "content": prompt})

    response_big = None
    idx = 0
    while response_big is None or idx == 10:
        response_big = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            n=1,
            max_tokens=max_tokens,
        )
        idx += 1

    if idx == 10:
        return "This sentence was not translated."

    response = response_big.choices[0].message.content

    return response
