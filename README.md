**Optimizing for Relevance and Diversity in RAG**

This project proposes a novel document utility function for retrieval in reasoning-intensive Retrieval-Augmented Generation (RAG) pipelines. The goal is to optimize both the relevance and diversity of retrieved documents without requiring a specialized retrieval model.

------------------------------

**The Challenge**
Reasoning-intensive queries, like those found in the BRIGHT Benchmark, often require synthesizing information from multiple documents. Standard retrieval methods can struggle to provide all the necessary context in a single pass. To address this, we use a Large Language Model (LLM) to rewrite and expand the initial query, ensuring a more comprehensive set of documents can be retrieved.

----------------------------

**Key Contributions**
This project investigates three core questions:

Prompt Engineering: Does a domain-adapted prompt, which explicitly defines the LLM's role and emphasizes nuance, outperform a general-purpose prompt template for query expansion?

Query Combination: When combining results from the original and expanded queries, which method yields better performance: a score-based or a rank-based approach?

Utility Function: How does the proposed document utility function, adapted from fundamental Information Retrieval theory, optimize for both relevance and diversity?

----------------------------

**Methodology**
We test our approach using both lexical models (BM25) and dense models (SBERT, OpenAI, Google) to retrieve documents. The reranking of documents is performed using the workflow that I proposed, reducing the need to train a separate retrieval model.

A detailed analysis of the results will be available in an upcoming Medium article.

Getting Started
Prerequisites

Python 3.8+

API keys for the services you plan to use (e.g., OpenAI, Google). You must provide these in a config.json file.

Installation

Clone this repository:

Bash
git clone [https://github.com/your-username/your-repository-name.git](https://github.com/FebrianiFR/uoe-dissertation.git)
cd uoe-dissertation
Install the required dependencies:

Bash
pip install -r requirements.txt
Data Setup

Due to file size limitations, the document embeddings for the BRIGHT Benchmark are not included in this repository.

Run the BRIGHT benchmark to generate the document embeddings.

Place the generated embeddings in the correct folder structure as follows:


```
📦 embeddings
├─ openai
│  ├─ doc_emb
│  └─ doc_ids
├─ google
└─ sbert
```




------------------------

Contact
If you encounter any issues or have questions, feel free to reach out to me at febriani.fitria@gmail.com.
