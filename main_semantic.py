from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv

load_dotenv()

# Load the document
superherofacts = TextLoader('superhero_facts.txt')

# Create the SemanticChunker
superherofacts_text_splitter = SemanticChunker(
    OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY")),
    number_of_chunks=10
)

# Load and split the text
chunks = superherofacts.load_and_split(text_splitter=superherofacts_text_splitter)

# Print the chunks
for i, chunk in enumerate(chunks, 1):
    print(f"Chunk {i}: {chunk.page_content}")

# Plot chunk sizes
chunk_sizes = [len(chunk.page_content) for chunk in chunks]
plt.figure(figsize=(10, 5))
plt.bar(range(1, len(chunk_sizes) + 1), chunk_sizes)
plt.title('Semantic Chunking (LangChain): Chunk Sizes')
plt.xlabel('Chunk Number')
plt.ylabel('Chunk Size (characters)')
plt.xlim(0.5, len(chunk_sizes) + 0.5) 
plt.xticks(range(1, len(chunk_sizes) + 1))  
plt.show()