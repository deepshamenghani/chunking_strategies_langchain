chunk_size = 150

with open('superhero_facts.txt', 'r') as file:
    data = file.read()

chunks = [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]

for i, chunk in enumerate(chunks, 1):
    print(f"Chunk {i}: {chunk}")

# Plot chunk sizes
import matplotlib.pyplot as plt

chunk_sizes = [len(chunk) for chunk in chunks]
plt.figure(figsize=(10, 5))
plt.bar(range(1, len(chunk_sizes) + 1), chunk_sizes)
plt.title('Fixed-size Chunking: Chunk Sizes')
plt.xlabel('Chunk Number')
plt.ylabel('Chunk Size (characters)')
plt.xlim(0.5, len(chunk_sizes) + 0.5) 
plt.xticks(range(1, len(chunk_sizes) + 1))  
plt.show()