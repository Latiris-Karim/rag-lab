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

class StandardChunker(Chunker):
    def __init__(self, file_path):
        self.file_path = file_path

    def chunk(self):
        chunks = []
        for file in os.listdir(self.file_path):
            full_path = os.path.join(self.file_path, file)
            if file.endswith('.md'):
                text = open(full_path, encoding='utf-8').read()
            else:
                text = md.convert(full_path).markdown
            res = rec_chunker.chunk(text)
            chunks.extend((file, chunk.text) for chunk in res)
        print(len(chunks))
        return chunks
    
class StandardVectorStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=os.getenv("db_path") + "chromadb_standard_rag")
        self.collection = self.client.get_or_create_collection(name="documents")

    def add_chunks(self, chunks):
        #chunk 0 = file name, chunk 1 = text
        for i, chunk in enumerate(chunks):
            embeddings = sentence_model.encode([chunk[1]]).tolist()
            self.collection.add(ids=str(i), embeddings=embeddings, documents=[chunk[1]], metadatas=[{"file": chunk[0]}])

    
if __name__ == "__main__":
    #chunking + embedding into vector store
    vector_store = StandardVectorStore()
    if vector_store.collection.count() == 0: #only chunk and embed if collection is empty
        chunker = StandardChunker(os.getenv('rag_files'))
        chunks = chunker.chunk()
        vector_store.add_chunks(chunks)

    #get context, query LLM, receive answer
    user = LLMInterface(Retriever(vector_store))
    answer = user.user_query("What is the main topic of the document?")
    print(answer)

