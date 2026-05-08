import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # supress TensorFlow warning about oneDNN optimizations 
from sentence_transformers import SentenceTransformer
from base_classes import Chunker, Retriever, LLMInterface
from markitdown import MarkItDown
import chromadb
from chonkie import RecursiveChunker


rec_chunker = RecursiveChunker.from_recipe("markdown", tokenizer="gpt2", chunk_size=300)
md = MarkItDown()
sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

class ContextualChunker(Chunker):
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

        print(len(chunks))
        return chunks
    
    def summary_for_chunk(self, document, chunks):
        summaries = []
        prompt = f"Summarize the main purpose of the chunk based on the document content. The summary should be a short sentence that explains the role this chunk plays in the overall document. Document: {document}\n\n"
        for chunk in chunks:
            res = self.user_interface.simple_query(prompt + f"Chunk: {chunk.text}")
            summaries.append(res)
        return summaries
            

class ContextualVectorStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=os.getenv("db_path") + "chromadb_contextual_rag")
        self.collection = self.client.get_or_create_collection(name="documents")

    def add_chunks(self, chunks):
        #chunk[0] = file name | chunk[1] = contextual summary | chunk[2] = chunk
        for i, chunk in enumerate(chunks):
            embeddings = sentence_model.encode([chunk[0] + chunk[1] + chunk[2]]).tolist()
            self.collection.add(ids=str(i), embeddings=embeddings, documents=[chunk[1] + chunk[2]], metadatas=[{"file": chunk[0]}])

if __name__ == "__main__":
    #chunking + embedding into vector store
    vector_store = ContextualVectorStore()
    user = LLMInterface(Retriever(vector_store))
    if vector_store.collection.count() == 0:
        chunker = ContextualChunker(os.getenv('rag_files'), user)
        chunks = chunker.chunk()
        vector_store.add_chunks(chunks)

    #get context, query LLM, receive answer
    answer = user.user_query("What is the main topic of the document?")
    print(answer)
