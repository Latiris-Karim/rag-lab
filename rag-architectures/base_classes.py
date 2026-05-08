from abc import ABC
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError
from FlagEmbedding import FlagReranker
from sentence_transformers import SentenceTransformer
import os

load_dotenv()

openai_client = OpenAI(api_key=os.getenv('API_KEY'), base_url="https://api.deepseek.com")
sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

class LLMInterface:
    def __init__(self, retriever):
        self.retriever = retriever

    def simple_query(self, query):
        try:
            response = openai_client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": query}],
                temperature=0.2,
            )
            return response.choices[0].message.content
        except OpenAIError as e:
            print(f"OpenAI API error: {e}")
            return "Sorry, I couldn't process your request at the moment."

    def user_query(self, query):
        context = self.retriever.get_context(query)
        prompt = f"Use the following context to answer the question:\n\n{context}\n\nQuestion: {query}\nAnswer:"

        try:
            response = openai_client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            return response.choices[0].message.content
        except OpenAIError as e:
            print(f"OpenAI API error: {e}")
            return "Sorry, I couldn't process your request at the moment."

class Chunker(ABC):
    def chunk(self):
        ...

class Retriever:
    def __init__(self, vector_store):
        self.collection = vector_store.collection
    
    def get_context(self, query, k=20):
        query_embedding = sentence_model.encode([query]).tolist()
        results = self.collection.query(query_embeddings=query_embedding, n_results=k)
        return results['documents'][0]
    

class Reranker:
    def __init__(self):
        self.reranker = FlagReranker('BAAI/bge-reranker-large', use_fp16=True)

    def rerank(self, query: str, candidates, k):
        #higher score = more relevant
        scores = self.reranker.compute_score([(query, candidate) for candidate in candidates])
        #plug in score with candidate/context and sort by score desc
        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        return [context for _, context in ranked[:k]]

if __name__ == "__main__":
    #reranker test
    reranker = Reranker()
    query = "What is the capital of France?"
    candidates = ["Berlin is the capital of Germany.", "Madrid is the capital of Spain.", "Paris is the capital of France."]
    result = reranker.rerank(query, candidates, k=1)
    print(result)
