from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
import re
import matplotlib.pyplot as plt

# Load the document
loader = TextLoader('superhero_facts.txt')
document = loader.load()[0]
data = document.page_content

# Level 1: Character names
def extract_character_names(text):
    return re.findall(r'\b[A-Z][a-z]+(?:-[A-Z][a-z]+)?\b', text)

# Level 2: Power summaries
power_splitter = CharacterTextSplitter(
    separator=".",
    chunk_size=50,
    chunk_overlap=0
)

# Level 3: Full bios
bio_splitter = RecursiveCharacterTextSplitter(
    separators=["\n"],
    chunk_size=1000,
    chunk_overlap=0
)

# Perform hierarchical chunking
level1_chunks = extract_character_names(data)
level2_chunks = power_splitter.split_text(data)
level3_chunks = bio_splitter.split_text(data)

# Print the hierarchical chunks
print("Level 1 (Character Names):")
for i, chunk in enumerate(level1_chunks, 1):
    print(f"  Chunk {i}: {chunk}")

print("\nLevel 2 (Power Summaries):")
for i, chunk in enumerate(level2_chunks, 1):
    print(f"  Chunk {i}: {chunk}")

print("\nLevel 3 (Full Bios):")
for i, chunk in enumerate(level3_chunks, 1):
    print(f"  Chunk {i}: {chunk[:100]}...") # Truncated for brevity

# Plot chunk sizes for each level
plt.figure(figsize=(15, 5))

plt.subplot(131)
plt.bar(range(1, len(level1_chunks) + 1), [len(chunk) for chunk in level1_chunks])
plt.title('Level 1: Character Names')
plt.xlabel('Chunk Number')
plt.ylabel('Chunk Size (characters)')

plt.subplot(132)
plt.bar(range(1, len(level2_chunks) + 1), [len(chunk) for chunk in level2_chunks])
plt.title('Level 2: Power Summaries')
plt.xlabel('Chunk Number')
plt.ylabel('Chunk Size (characters)')

plt.subplot(133)
plt.bar(range(1, len(level3_chunks) + 1), [len(chunk) for chunk in level3_chunks])
plt.title('Level 3: Full Bios')
plt.xlabel('Chunk Number')
plt.ylabel('Chunk Size (characters)')

plt.tight_layout()
plt.show()