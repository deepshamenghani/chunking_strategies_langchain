from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
import matplotlib.pyplot as plt

# Load the document
superherofacts = TextLoader('superhero_facts.txt')

# Create the CharacterTextSplitter
text_splitter = CharacterTextSplitter(
    separator=" ",
    chunk_size=150,
    chunk_overlap=20,
    length_function=len
)

# Load and split the text
chunks = superherofacts.load_and_split(text_splitter=text_splitter)

# Print the chunks
for i, chunk in enumerate(chunks, 1):
    print(f"Chunk {i}: {chunk.page_content}")

# Plot chunk sizes
chunk_sizes = [len(chunk.page_content) for chunk in chunks]
plt.figure(figsize=(10, 5))
plt.bar(range(1, len(chunk_sizes) + 1), chunk_sizes)
plt.title('Character-based Chunking (LangChain): Chunk Sizes')
plt.xlabel('Chunk Number')
plt.ylabel('Chunk Size (characters)')
plt.xlim(0.5, len(chunk_sizes) + 0.5) 
plt.xticks(range(1, len(chunk_sizes) + 1))  
plt.show()