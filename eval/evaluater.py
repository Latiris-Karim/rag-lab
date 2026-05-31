import os
import sys
import json
sys.path.append('rag-architectures')
from hybrid_rag import HybridVectorStore, BM25, HybridRetriever
from contextual_rag import ContextualVectorStore
from standard_rag import StandardVectorStore
from base_classes import LLMInterface, Retriever, Reranker




class TestSetGenerator:
    def __init__(self, chunks):
        self.chunks = chunks
        self.llm = LLMInterface()

    def generate_test_set(self):
        test_set = []
        for id, chunk in self.chunks:
            question = self._generate_question_for_chunk(chunk)
            test_set.append({"question": question, "id": id})
        with open("test_set.json", "w") as f:
            json.dump(test_set, f, indent = 2)
        return test_set

    def _generate_question_for_chunk(self, chunk):
        prompt = f"Generate a question that can be answered by the following chunk of text: {chunk}"
        question = self.llm.simple_query(prompt)
        return question


def run_evaluation(testset, retrievers):
    results = {retriever: 0 for retriever in retrievers}
    for name, retriever in retrievers.items():
        for item in testset:
            question = item["question"]
            chunk_id = item["id"]
            ids = retriever.get_ids(question)
            if str(chunk_id) in ids:
                results[name] += 1
    return results

if __name__ == "__main__":
    hybrid_vector_store = HybridVectorStore()

    # step 1, generate test set. pairs of (question, chunk id)
    if "test_set.json" not in os.listdir():
        data = hybrid_vector_store.collection.get()
        documents = data['documents']
        ids = data['ids']
        
        test_set = TestSetGenerator(list(zip(ids, documents))).generate_test_set()
    else: 
        with open("test_set.json", "r") as f:
            test_set = json.load(f)
    
    #step 2, evaluate retrieval on the different RAG architectures
    naive_RAG_store = StandardVectorStore()
    contextual_RAG_store = ContextualVectorStore()

    hybrid_RAG = HybridRetriever(hybrid_vector_store, BM25(), Reranker())
    contextual_RAG = Retriever(contextual_RAG_store)
    standard_RAG = Retriever(naive_RAG_store)

    ranking = {"Hybrid": hybrid_RAG, "Contextual": contextual_RAG, "Standard": standard_RAG}

    results = run_evaluation(test_set, ranking)
    for name, hits in results.items():
        print(f"{name}: {hits / len(test_set) * 100:.1f}%")
