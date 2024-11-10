import json
import pathlib
import threading
import vllm
import wordllama


class Chatbot:
    def __init__(
        self,
        knowledge_base: str,
        sampling_params="",
        reader_model_id: str = "",
        embedder_model_id="",
        reranker_model_id="",
        use_decoder_as_embedder=False,
    ) -> None:
        self.knowledge_base = knowledge_base
        self.reader_model_id = reader_model_id
        self.wl = None
        self.llm = None
        self.chunks = None
        self.is_loaded = False  # Flag to check if loading is done
        self._loading_thread = None  # Hold the loading thread reference

        self.sampling_params = vllm.SamplingParams(
            **json.loads(pathlib.Path(sampling_params).read_text())
        )

        if "gemma" not in self.reader_model_id.lower():
            self.messages = [
                {
                    "role": "system",
                    "content": "You will be given documents and a question. Your task is to answer the question using these documents. Be factual and only use information from the context to answer the questions. Be concise in your answers, not more than one sentence.",
                },
            ]
        else:
            self.messages = [
                {
                    "role": "user",
                    "content": "You will be given documents and a question. Your task is to answer the question using these documents. Be factual and only use information from the context to answer the questions. Be concise in your answers, not more than one sentence.",
                },
                {
                    "role": "assistant",
                    "content": "Okay! Send me the context and the question.",
                },
            ]

        # Start loading in a background thread
        self._loading_thread = threading.Thread(target=self._load_resources)
        self._loading_thread.start()

    def _load_resources(self) -> None:
        # Loading WordLlama embeddings
        self.wl = wordllama.WordLlama.load(trunc_dim=256)

        print(self.reader_model_id)
        # Loading model
        self.llm = vllm.LLM(
            model=self.reader_model_id,
            max_model_len=4096,
            trust_remote_code=True,
        )

        # Semantic chunking
        self.chunks = self.wl.split(self.knowledge_base, target_size=256)

        # Set the flag to True once loading is complete
        self.is_loaded = True

    def __str__(self):
        return f"""Reader Model: {self.reader_model_id}
Embedder model: wordllama
Reranker model: wordllama
"""

    def wait_for_load(self) -> None:
        if not self.is_loaded:
            self._loading_thread.join()

    def shutdown(self) -> None:
        if self._loading_thread and self._loading_thread.is_alive():
            print("Waiting for resources to finish loading before shutting down...")
            self._loading_thread.join()  # Block until the thread is complete

    def retrieve(self, question: str) -> list[tuple[str, float]]:
        self.wait_for_load()

        top_docs = self.wl.topk(question, self.chunks, k=6)
        return [
            (sent, sim)
            for sent, sim in self.wl.rank(question, top_docs, sort=True)
            if sim > 0.1
        ]

    def send_question(
        self,
        question: str,
    ) -> tuple[str, list[tuple[str, float]]]:
        self.wait_for_load()
        ranked = self.retrieve(question)

        context = "\n".join(
            f"""DOCUMENT SIMILARITY: 

    {sim:.2f}

    DOCUMENT TEXT:

    {sent}
    """
            for sent, sim in ranked
        )
        messages_ = self.messages.copy()
        messages_.append(
            {
                "role": "user",
                "content": f"""CONTEXT:

        {context}

        QUESTION:

        {question}""",
            }
        )

        res = self.llm.chat(messages_, self.sampling_params)

        return res[0].outputs[0].text, "\n".join(
            (f"Sim: {sim:.2f} - {doc}" for doc, sim in ranked[:6])
        )


if __name__ == "__main__":
    s = pathlib.Path("data/orientation.md").read_text()
    c = Chatbot(
        knowledge_base=s,
        reader_model_id="Qwen/Qwen2-1.5B-Instruct",
        sampling_params="cfg/qwen_25_7b.json",
        embedder_model_id="sentence-transformers/all-MiniLM-L6-v2",
        reranker_model_id="BAAI/bge-reranker-base",
        use_decoder_as_embedder=False,
    )

    question = "What should students do if they have questions or technical issues with the course material?"
    response, context = c.send_question(question)

    print("\n\n>>>" + response + "<<<")
    print()
    print(context)
    print()
    print(c)
