from langchain.text_splitter import TokenTextSplitter
from langchain_community.document_loaders import TextLoader
import matplotlib.pyplot as plt

# Initialize the TokenTextSplitter
token_text_splitter = TokenTextSplitter(
    chunk_size=150,  
    chunk_overlap=0 
)

# Load and split the document
loader = TextLoader("superhero_facts.txt")
token_docs = loader.load_and_split(text_splitter=token_text_splitter)

# Print token-based chunks
print("Token-based Chunks:")
for doc in token_docs:
    print(doc.page_content)
    print("\n")

# Collect token-based chunk sizes
token_chunk_sizes = [len(doc.page_content) for doc in token_docs]

# Create a bar plot for token-based chunks
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(token_chunk_sizes) + 1), token_chunk_sizes)
plt.xlabel('Chunk Number')
plt.ylabel('Chunk Size (characters)')
plt.title('Token-based Chunk Sizes after Splitting')
plt.xlim(0.5, len(token_chunk_sizes) + 0.5)
plt.xticks(range(1, len(token_chunk_sizes) + 1))
plt.show()