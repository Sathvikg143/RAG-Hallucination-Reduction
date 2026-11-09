from modules.pdf_loader import PDFLoader
from modules.chunker import TextChunker
from modules.embeddings import EmbeddingGenerator
from modules.vector_store import VectorStore

loader = PDFLoader("data")
documents = loader.load_documents()

print(f"\nLoaded {len(documents)} pages.\n")


chunker = TextChunker()

chunks = chunker.create_chunks(documents)

print(f"Created {len(chunks)} chunks.\n")


for chunk in chunks[:5]:


    print(f"Chunk ID : {chunk['chunk_id']}")
    print(f"Source   : {chunk['source']}")
    print(f"Page     : {chunk['page']}")

    print()

    print(chunk["text"])

    print()
    lengths = [len(chunk["text"]) for chunk in chunks]

print("\nChunk Statistics")
print("-" * 30)
print(f"Minimum : {min(lengths)}")
print(f"Maximum : {max(lengths)}")
print(f"Average : {sum(lengths)//len(lengths)}")
embedding_model = EmbeddingGenerator()

embeddings = embedding_model.generate_embeddings(chunks)

print("\nEmbedding Shape")
print("-"*30)
print(embeddings.shape)

print("\nFirst 10 values of first embedding")

print(embeddings[0][:10])

vector_db = VectorStore()

vector_db.build(
    embeddings,
    chunks
)

vector_db.save()

print("\nFAISS Index Created Successfully!")

print(f"Vectors Stored : {vector_db.index.ntotal}")