import json
import pathlib

import faiss
import numpy as np
import torch
import torch.nn.functional as F
import transformers
import tqdm
import vllm

from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings


class Chatbot:
    def __init__(
        self,
        knowledge_base: str,
        reader_model_id: str = "",
        sampling_params: str = "",
        embedder_model_id: str = "",
        reranker_model_id: str = "",
        use_decoder_as_embedder: bool = False,
    ) -> None:
        self.reader_model_id = reader_model_id
        self.embedder_model_id = embedder_model_id
        self.reranker_model_id = reranker_model_id

        # Dirty hack to find out if the model supports system role. TODO: rewrite with tokenizers.
        if "gemma" not in reader_model_id:
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

        self.reader_llm = vllm.LLM(
            model=self.reader_model_id,
            max_model_len=4096,
            trust_remote_code=True,
        )

        self.sampling_params = vllm.SamplingParams(
            **json.loads(pathlib.Path(sampling_params).read_text())
        )

        self.embedder_model = transformers.AutoModel.from_pretrained(
            self.embedder_model_id,
            trust_remote_code=True,
        )
        self.embedder_model.eval()
        self.embedder_tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.embedder_model_id,
        )
        self.use_decoder_as_embedder = use_decoder_as_embedder

        self.reranker_model = (
            transformers.AutoModelForSequenceClassification.from_pretrained(
                self.reranker_model_id,
                trust_remote_code=True,
            )
        )
        self.reranker_model.eval()
        self.reranker_tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.reranker_model_id
        )

        self.knowledge_base, self.embedding_index = self.build_database(knowledge_base)

    def __str__(self):
        return f"""Reader Model: {self.reader_model_id}
Embedder model: {self.embedder_model_id}
Reranker model: {self.reranker_model_id}
"""

    def embed(self, query):
        def average_pool(
            last_hidden_states,
            attention_mask,
        ):
            last_hidden = last_hidden_states.masked_fill(
                ~attention_mask[..., None].bool(), 0.0
            )
            return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

        def last_token_pool(
            last_hidden_states,
            attention_mask,
        ):
            left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
            if left_padding:
                return last_hidden_states[:, -1]
            else:
                sequence_lengths = attention_mask.sum(dim=1) - 1
                batch_size = last_hidden_states.shape[0]
                return last_hidden_states[
                    torch.arange(batch_size, device=last_hidden_states.device),
                    sequence_lengths,
                ]

        with torch.no_grad():
            input_ids = self.embedder_tokenizer(
                [query],
                max_length=512,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            outputs = self.embedder_model(**input_ids)

            pooling_function = (
                last_token_pool if self.use_decoder_as_embedder else average_pool
            )
            embeddings = pooling_function(
                outputs.last_hidden_state, input_ids["attention_mask"]
            )
            embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings

    def build_database(self, knowledge_base: str) -> tuple:
        # Getting embedding size for the faiss
        emb_size = len(self.embed("test embedding")[0])
        embedding_db = np.empty((0, emb_size), dtype=np.float32)

        model_name = self.embedder_model_id
        model_kwargs = {"device": "cuda"}
        encode_kwargs = {"normalize_embeddings": False}
        hf = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )

        text_splitter = SemanticChunker(hf)
        docs = text_splitter.create_documents([knowledge_base])

        idx_to_str = dict()

        for idx, chunk in tqdm.tqdm(enumerate(docs)):
            chunk = chunk.page_content
            embedding = self.embed(chunk)
            embedding_db = np.append(embedding_db, embedding, axis=0)

            idx_to_str[idx] = chunk

        embedding_index = faiss.IndexFlatL2(emb_size)
        embedding_index.add(embedding_db)

        return idx_to_str, embedding_index

    def rerank(
        self,
        question: str,
        retrieved_documents: list[str],
    ) -> list[tuple[float, str]]:
        # TODO: Same for LLM-based rerankers
        pairs = [(question, document) for document in retrieved_documents]

        with torch.no_grad():
            inputs = self.reranker_tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            )
            scores = (
                self.reranker_model(**inputs, return_dict=True)
                .logits.view(
                    -1,
                )
                .float()
            )

        results = [(score, document) for score, (_, document) in zip(scores, pairs)]
        results = sorted(results, key=lambda x: x[0], reverse=True)
        return results

    def retrieve(self, question: str):
        _, sid = self.embedding_index.search(self.embed(question), 10)
        reranked = self.rerank(question, [self.knowledge_base[_] for _ in sid[0]])
        return [(sent, sim) for sim, sent in reranked if sim > 0.1]

    def send_question(
        self,
        question: str,
    ) -> tuple[tuple[str, list[tuple[str, float]]], str]:
        ranked = self.retrieve(question)
        context = "\n".join(
            f"""DOCUMENT SIMILARITY: 

    {sim:.2f}

    DOCUMENT TEXT:

    {sent}
    """
            for sent, sim in ranked[:6]
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

        res = self.reader_llm.chat(messages_, self.sampling_params)

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
