# ColBERTv2 vs Bi-Encoder Retrieval Benchmark

This project is a study on how to make Retrieval-Augmented Generation (RAG) better. We use the KILT NaturalQuestions dataset to test different ways of cutting text into pieces (chunking) and different ways of finding the right information (retrieval). We compare a standard Bi-Encoder with the more complex ColBERTv2 model. We also added a Knowledge Graph and an LLM generation step to finish the full RAG process.

## Setup

To run this project, you need Python 3.10 or higher. You will also need some extra disk space for the Wikipedia data and the search indexes.

First, create and start a virtual environment:
python -m venv .venv

On Windows:
.venv\Scripts\Activate.ps1

On Mac or Linux:
source .venv/bin/activate

Next, install the needed packages:
pip install -r requirements.txt

Finally, download the spaCy model for entity extraction:
python -m spacy download en_core_web_sm

## How to Run

The main way to run this project is by using the Jupyter Notebook. Open it with this command:
jupyter notebook notebooks/run_benchmark.ipynb

You should run the notebook one section at a time. It will download the data, cut the text into chunks, build the search indexes, and run the evaluation. The results like charts and tables will be saved in the "results" folder.

## Configuration and Hardware

We provide two main settings files in the "configs" folder. "quick_ablation.yaml" is small and fast. It is best for testing on a normal laptop or when you want to save API tokens. "experiment_config.yaml" is much larger and needs a strong GPU with a lot of VRAM.

ColBERTv2 needs a lot of memory and special C++ tools to run. If you are on a Windows laptop with less than 8GB of VRAM, the full version might fail. For these cases, we recommend using the smaller settings or the "mock" version of the retriever to see how the logic works without crashing your computer.

## Cutting Text (Chunking)

We tested several ways to cut the Wikipedia text. "Paragraph" chunking is our simple starting point. "Sentence window" groups a fixed number of sentences together. Our best method is "Adaptive Sentence" chunking. It looks at the text sentence by sentence and decides where to stop based on the word count and keywords. We also have "Semantic Similarity" chunking, which uses math to find where the topic changes and starts a new chunk there.

## Knowledge Graph and Hybrid Search

We built a simple Knowledge Graph to help find information. It finds important names or places (entities) in the text and links them together. We use IDF weights to make sure common words don't ruin the search results. 

The "Hybrid" search combines the Knowledge Graph results with the vector search results using a method called Reciprocal Rank Fusion (RRF). Our tests show that combining ColBERT with the Knowledge Graph works best for questions that mention many different entities.

## LLM Generation

Unlike the early versions of this project, we now have a full RAG system. After finding the best text chunks, the system sends them to a Llama 3 (via Groq API). The model uses only the provided text to answer the user's question. This makes sure the answer is based on real facts from the data.

## Project Structure

The "src" folder contains all the Python code for the data pipeline, the two types of retrievers, the knowledge graph, and the generation logic. The "notebooks" folder has the main file to run everything. All the settings are in the "configs" folder. When you run the code, all logs and images will appear in the "results" folder.

## Final Findings

Our experiments show that ColBERTv2 is more accurate than the Bi-Encoder, but it is also much slower and uses more memory. Adaptive chunking is very important because it keeps the context together better than fixed-size chunks. The strongest overall result comes from the hybrid system that uses both ColBERT and the Knowledge Graph.
