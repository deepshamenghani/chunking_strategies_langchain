from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
import matplotlib.pyplot as plt

# Load the document
superherofacts = TextLoader('superhero_facts.txt')
superherofacts_document = superherofacts.load()[0]
superherofacts_data = superherofacts_document.page_content

# Level 1: Character names
def extract_character_names(text):
    paragraphs = text.split('\n')
    return [para.split()[0:2] for para in paragraphs if para.strip()]

# Level 2: Power summaries
power_splitter = CharacterTextSplitter(
    separator=".",
    chunk_size=150,
    chunk_overlap=0
)

# Level 3: Full bios
bio_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " "],
    chunk_size=800,
    chunk_overlap=200,
    length_function=len
)

# Perform hierarchical chunking
level1_chunks_characternames = extract_character_names(superherofacts_data)
level2_chunks_powersummary = power_splitter.split_text(superherofacts_data)
level3_chunks_characterbio = bio_splitter.split_text(superherofacts_data)

# Print the hierarchical chunks
print("Level 1 (Character Names):")
for i, chunk in enumerate(level1_chunks_characternames, 1):
    print(f"  Chunk {i}: {' '.join(chunk)}")

print("\nLevel 2 (Power Summaries):")
for i, chunk in enumerate(level2_chunks_powersummary, 1):
    print(f"  Chunk {i}: {chunk}")

print("\nLevel 3 (Full Bios):")
for i, chunk in enumerate(level3_chunks_characterbio, 1):
    print(f"  Chunk {i}: {chunk[:100]}...") # Truncated for brevity

# Plot chunk sizes for each level
plt.figure(figsize=(15, 5))

plt.subplot(131)
plt.bar(range(1, len(level1_chunks_characternames) + 1), [len(' '.join(chunk)) for chunk in level1_chunks_characternames])
plt.title('Level 1: Character Names')
plt.xlabel('Chunk Number')
plt.ylabel('Chunk Size (characters)')

plt.subplot(132)
plt.bar(range(1, len(level2_chunks_powersummary) + 1), [len(chunk) for chunk in level2_chunks_powersummary])
plt.title('Level 2: Power Summaries')
plt.xlabel('Chunk Number')
plt.ylabel('Chunk Size (characters)')

plt.subplot(133)
plt.bar(range(1, len(level3_chunks_characterbio) + 1), [len(chunk) for chunk in level3_chunks_characterbio])
plt.title('Level 3: Full Bios')
plt.xlabel('Chunk Number')
plt.ylabel('Chunk Size (characters)')

plt.tight_layout()
plt.show()