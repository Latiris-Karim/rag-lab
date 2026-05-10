import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # supress TensorFlow warning about oneDNN optimizations 
from sentence_transformers import SentenceTransformer
from base_classes import Chunker, Retriever, LLMInterface, Reranker
from markitdown import MarkItDown
import chromadb
from chonkie import RecursiveChunker
from rank_bm25 import BM25Okapi
import pickle

rec_chunker = RecursiveChunker.from_recipe("markdown", tokenizer="gpt2", chunk_size=300)
md = MarkItDown()
sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

#is the same as ContextualChunker
class HybridChunker(Chunker):
    def __init__(self, file_path, user_interface:LLMInterface):
        self.file_path = file_path
        self.user_interface = user_interface

    def chunk(self):
        #add file name + a short sentence expaining the role this chunk plays in the document + chunk 
        chunks = []
        for file in os.listdir(self.file_path):
            full_path = os.path.join(self.file_path, file)
            if file.endswith('.md'):
                text = open(full_path, encoding='utf-8').read()
            else:
                text = md.convert(full_path).markdown
            res = rec_chunker.chunk(text)
            contexualsummary = self.summary_for_chunk(text, res) 

            chunks.extend((file, contexualsummary[chunk], res[chunk].text) for chunk in range(len(res)))
        return chunks
    
    def summary_for_chunk(self, document, chunks):
        summaries = []
        prompt = f"Summarize the main purpose of the chunk based on the document content. The summary should be a short sentence that explains the role this chunk plays in the overall document. Document: {document}\n\n"
        for chunk in chunks:
            res = self.user_interface.simple_query(prompt + f"Chunk: {chunk.text}")
            summaries.append(res)
        return summaries
            
#is the same as ContextualVectorStore
class HybridVectorStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=os.getenv("db_path") + "chromadb_hybrid_rag")
        self.collection = self.client.get_or_create_collection(name="documents")

    def add_chunks(self, chunks):
        #chunk[0] = file name | chunk[1] = contextual summary | chunk[2] = chunk
        for i, chunk in enumerate(chunks):
            embeddings = sentence_model.encode([chunk[0] + chunk[1] + chunk[2]]).tolist()
            self.collection.add(ids=str(i), embeddings=embeddings, documents=[chunk[1] + chunk[2]], metadatas=[{"file": chunk[0]}])


class BM25:
    def __init__(self):
        if os.path.exists("bm25_model.pkl"):
            model_data = pickle.load(open("bm25_model.pkl", "rb"))
            self.bm25 = model_data["bm25"]
            self.chunk_list = model_data["texts"]

    def retrieve(self, query):
        query = self.remove_non_alphanumeric(query.lower())
        tokenized_query = self.tokenize_query(query)
        relevant_docs = self.bm25.get_top_n(tokenized_query, self.chunk_list, n=20)
        return relevant_docs

    def index_chunks(self, corpus):
        self.tokenized_corpus = []
        self.chunk_list = []
        for chunk in corpus: 
            chunk_words = chunk[2].split(" ") #chunk[2] is the chunk text
            chunk_words = [self.remove_non_alphanumeric(word) for word in chunk_words]
            self.tokenized_corpus.append(chunk_words)
            self.chunk_list.append(chunk[1] + chunk[2]) #same format as vector documents

        #calculates term frequency per word for each chunk and inverse document frequency per word across all chunks
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        pickle.dump({"bm25": self.bm25, "texts": self.chunk_list}, open("bm25_model.pkl", "wb"))
        
    def tokenize_query(self, query):
        #returns query word for word for BM25 algorithm
        return query.split(" ")
    
    def remove_non_alphanumeric(self, text):
        #remove all non alphanumeric characters and make all words lowercase to improve BM25 performance
        return ''.join(e for e in text.lower() if e.isalnum() or e.isspace())


if __name__ == "__main__":
    vector_store = HybridVectorStore()
    retriever = Retriever(vector_store)
    user = LLMInterface(retriever)
    bm25 = BM25()
    reranker = Reranker()

     #chunking + embedding into vector store
    if vector_store.collection.count() == 0:
        chunker = HybridChunker(os.getenv('rag_files'), user)
        chunks = chunker.chunk()
        bm25.index_chunks(chunks)
        vector_store.add_chunks(chunks)

    #user query flow
    query= "What is the main topic of the document?"
    
    #get context from bm25 and vector store
    bm25_context = bm25.retrieve(query)
    emb_context = retriever.get_context(query)
    print("BM25 retrieved documents:")
    print((bm25_context))
    print("Vector store context:")
    print(type(emb_context))

    #compares bm25 and emb contexts and reranks in relevance to the query 
    #computes a top 5 list of the (currently) 20 bm25 and 20 embedding context chunks
    final_context= reranker.rerank(query, bm25_context + emb_context, k=5)
    print("Final context after reranking:")
    print(final_context)
    
