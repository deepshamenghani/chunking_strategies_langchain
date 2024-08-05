# Chunking Strategies for RAG Systems Using LangChain

This repository demonstrates various text chunking strategies for Retrieval-Augmented Generation (RAG) systems using LangChain. It showcases different approaches to splitting text for effective information retrieval and processing.

## Project Structure

* `main_fixed.py`: Implementation of fixed-size chunking
* `main_character.py`: Implementation of character-based chunking
* `main_token.py`: Implementation of token-based chunking
* `main_recursive.py`: Implementation of recursive chunking
* `main_paragraph.py`: Implementation of paragraph-based chunking
* `main_semantic.py`: Implementation of semantic chunking
* `main_hierarchical.py`: Implementation of hierarchical chunking
* `superhero_facts.txt`: Text file containing superhero facts used for demonstrations
* `.gitignore`: Specifies intentionally untracked files to ignore

## Features

* Multiple chunking strategies implementation
* Integration with LangChain's text splitters
* Demonstration of each strategy using superhero facts dataset
* Visualization of chunk sizes (where applicable)
* Utilization of OpenAI's embeddings for semantic chunking

## Installation

1. Clone this repository
2. Install required packages: pip install langchain langchain_openai langchain_community langchain_experimental python-dotenv matplotlib
3. Set up your OpenAI API key in a `.env` file: OPENAI_API_KEY=your_api_key_here

## Usage

Run each chunking strategy demonstration:
python main_fixed.py
python main_character.py
python main_token.py
python main_recursive.py
python main_paragraph.py
python main_semantic.py
python main_hierarchical.py

## How It Works

This project demonstrates various chunking strategies:

1. **Fixed-size Chunking**: Splits text into chunks of a predetermined size
2. **Character-based Chunking**: Splits text based on character count with user-defined break points
3. **Token-based Chunking**: Splits text based on the number of tokens
4. **Recursive Chunking**: Uses a list of separators to split text hierarchically
5. **Paragraph-based Chunking**: Keeps entire paragraphs together as single units
6. **Semantic Chunking**: Divides text based on its semantic content
7. **Hierarchical Chunking**: Creates multiple levels of text divisions

Each strategy is implemented in its own Python file, demonstrating the chunking process and outputting the results.

## Key Components

* **CharacterTextSplitter**: Splits text based on characters
* **TokenTextSplitter**: Splits text based on tokens
* **RecursiveCharacterTextSplitter**: Splits text recursively using multiple separators
* **SemanticChunker**: Splits text based on semantic similarity (requires OpenAI API)

## Customization

Feel free to modify the code to experiment with:
* Different chunk sizes
* Various separator strategies
* Custom splitting logic
* Different datasets

For a detailed explanation of each strategy, please refer to the associated blog post: [There's more than one way to chunk a RAG](https://medium.com/@menghani.deepsha)