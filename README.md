# Public-Sector Agentic RAG Assistant

An experimental Retrieval-Augmented Generation (RAG) system for grounded, traceable, and evidence-aware public-sector information assistance.

Developed as part of a University of Alberta Master of Modelling, Data and Predictions (MDP) capstone project, this prototype investigates how retrieval, reranking, query routing, evidence assessment, citation traceability, clarification, and abstention can be combined to support more reliable use of large language models in public-sector information environments.

The prototype uses a small corpus of publicly available Government of Alberta information covering vehicle registration, motor vehicle information, and land titles.

> **Research prototype:** This system is not an official Government of Alberta service and should not be used as a substitute for authoritative government or legal advice.

---

## Project Objectives

Public-sector information assistants must do more than generate fluent responses. They should be able to:

- ground answers in authoritative source material;
- distinguish supported questions from unsupported ones;
- request clarification when a question is ambiguous;
- avoid answering questions outside the supported service scope;
- abstain when retrieved evidence is insufficient;
- preserve traceability between generated claims and source evidence.

This prototype implements and evaluates a pipeline designed around these requirements.

---

## System Architecture

The end-to-end pipeline is:

```text
User Query
    |
    v
Query Processor
    |
    +---- Ambiguous ----------> Clarification
    |
    +---- Out of Scope -------> Scope Response
    |
    v
Semantic Retrieval
BGE-small-en-v1.5
    |
    v
Chroma Vector Store
    |
    v
Initial Candidate Retrieval
    |
    v
CrossEncoder Reranking
ms-marco-MiniLM-L-6-v2
    |
    v
Evidence Sufficiency Gate
    |
    +---- Insufficient -------> Abstention
    |
    v
Grounded Answer Generation
Gemini
    |
    v
Chunk-Level Citations
    |
    v
Final Response
```

The orchestration layer coordinates these components and determines whether the system should clarify, reject an out-of-scope request, retrieve evidence, abstain, or generate a grounded answer.

---

## Core Components

### 1. Public-Sector Corpus

The prototype corpus contains publicly available Government of Alberta material across three service areas:

- vehicle registration;
- motor vehicle information;
- land titles.

Both HTML pages and selected PDF documents are processed into structured documents and chunks.

The current vector collection contains **120 chunks**.

### 2. Semantic Retrieval

Document chunks are embedded using:

`BAAI/bge-small-en-v1.5`

Embeddings are stored and queried using **ChromaDB**.

The first retrieval stage returns candidate evidence based on semantic similarity.

### 3. CrossEncoder Reranking

Initial candidates are reranked using:

`cross-encoder/ms-marco-MiniLM-L-6-v2`

The CrossEncoder jointly evaluates the user query and each candidate passage to produce a more precise relevance ranking.

### 4. Query Processing and Routing

Before retrieval, the query processor determines whether the request:

- belongs to a supported prototype domain;
- requires clarification because it is ambiguous; or
- falls outside the current prototype scope.

This prevents every user input from automatically reaching the generation stage.

### 5. Evidence Sufficiency Gate

Retrieved evidence is evaluated before an answer is generated.

When the available evidence is not sufficiently relevant, the system abstains rather than attempting to fill information gaps using unsupported model knowledge.

### 6. Grounded Generation

For sufficiently supported questions, retrieved evidence is supplied to Gemini with instructions to answer only from the provided context.

The generator is instructed not to invent missing requirements, procedures, forms, deadlines, fees, exceptions, or legal conclusions.

### 7. Citation Traceability

Evidence chunks have stable identifiers such as:

```text
VR-02-CH-002
LT-05-CH-001
MVI-02-CH-001
```

Generated responses cite these chunk identifiers, allowing claims to be traced back to retrieved evidence and its original source URL.

### 8. Abstention

The system supports explicit abstention when the corpus does not contain enough evidence to answer a question reliably.

For example, a question can be within the general land-title domain while still asking for information that is not contained in the available corpus. In that case, the system can recognize the domain but decline to generate an unsupported answer.

---

## Evaluation

The final prototype was evaluated using a **30-question end-to-end test set** covering four behavioural categories.

| Evaluation Category | Tests | Passed | Accuracy |
|---|---:|---:|---:|
| Supported questions | 15 | 15 | 100% |
| Ambiguous questions | 5 | 5 | 100% |
| Out-of-scope questions | 5 | 5 | 100% |
| Insufficient-evidence questions | 5 | 5 | 100% |
| **Overall** | **30** | **30** | **100%** |

The expected status distribution was:

```text
answered:      15
clarify:        5
out_of_scope:   5
abstain:        5
```

The observed distribution matched the expected distribution exactly.

**Important:** The 100% result refers only to this fixed 30-question prototype evaluation set. It should not be interpreted as evidence that the system will achieve 100% accuracy on unseen queries or in production.

The evaluation questions and generated results are available in:

```text
data/evaluation/evaluation_questions.csv
data/evaluation/evaluation_results.csv
```

---

## Example Behaviours

### Supported Question

```text
What information do I need to transfer land?
```

The system retrieves and reranks relevant land-title evidence and generates a grounded answer with chunk-level citations.

### Ambiguous Question

```text
How do I transfer it?
```

The system does not assume what "it" refers to and asks the user to clarify the intended service.

### Out-of-Scope Question

```text
How do I apply for an Alberta health card?
```

The system identifies that the request is outside the three service domains represented in the prototype.

### Insufficient Evidence

```text
How long does a land title transfer take to process?
```

Although the query relates to land titles, the available evidence does not sufficiently support an answer, so the system abstains.

---

## User Interface

A Streamlit interface provides a chatbot-style front end for interacting with the prototype.

The interface connects to the same end-to-end orchestration pipeline used during evaluation.

Run the interface with:

```bash
streamlit run app.py
```

---

## Repository Structure

```text
public-sector-agentic-rag-mdp-capstone/
|
|-- app.py
|-- README.md
|-- requirements.txt
|
|-- data/
|   |-- evaluation/
|   |-- processed/
|   |-- raw/
|   `-- vector_store/        # local generated vector database
|
|-- docs/
|   `-- corpus_manifest.csv
|
`-- src/
    |-- agent.py
    |-- answer_generator.py
    |-- query_processor.py
    |-- query_router.py
    |-- retrieval_pipeline.py
    |-- retrieve_bge.py
    |-- evidence_gate.py
    |-- build_vector_store_bge.py
    |-- evaluate_agent.py
    |-- evaluate_retrieval_bge.py
    |-- evaluate_retrieval_rerank.py
    `-- supporting processing, validation, and test scripts
```

---

## Installation

Clone the repository and install the required Python packages:

```bash
pip install -r requirements.txt
```

Create a `.env` file inside `src/` containing:

```text
GEMINI_API_KEY=your_api_key_here
```

The `.env` file is excluded from version control.

The local Chroma vector store is also excluded from version control and can be rebuilt from the processed corpus.

---

## Running the Prototype

To launch the Streamlit interface:

```bash
streamlit run app.py
```

To run the end-to-end evaluation:

```bash
python src/evaluate_agent.py
```

To run the retrieval and reranking evaluation:

```bash
python src/evaluate_retrieval_rerank.py
```

---

## Technology Stack

- **Python**
- **Streamlit** — prototype user interface
- **ChromaDB** — vector storage and retrieval
- **Sentence Transformers**
- **BAAI/bge-small-en-v1.5** — embedding model
- **cross-encoder/ms-marco-MiniLM-L-6-v2** — reranker
- **Google Gemini API** — grounded answer generation
- **pypdf / BeautifulSoup** — source-document processing

---

## Limitations

This prototype intentionally has a narrow scope.

Key limitations include:

- a small curated corpus covering only three Alberta public-service domains;
- a small, manually defined evaluation set;
- rule-based domain and ambiguity detection;
- evidence thresholds calibrated for the prototype corpus;
- no claim of generalization to unseen public-service domains;
- no production security, authentication, monitoring, or scalability layer;
- dependence on the quality and currency of the underlying source documents;
- generated answers remain subject to the limitations of the underlying language model.

Future work could evaluate the architecture on a larger corpus, introduce learned or LLM-based routing, test more adversarial and paraphrased queries, improve evidence-sufficiency calibration, and evaluate answer-level factuality and citation correctness at larger scale.

---

## Capstone Context

This repository supports an MDP capstone investigating the design of an evidence-grounded, agentic RAG architecture for public-sector information assistance.

The implementation is intended as a proof of concept demonstrating how retrieval, reranking, evidence gating, grounded generation, traceability, clarification, and abstention can be integrated into a single end-to-end workflow.

---

## License

See `LICENSE` for repository licensing information.