# MatScribe: Agentic Multimodal RAG for Materials Discovery

## Overview
MatScribe is an automated Retrieval-Augmented Generation (RAG) pipeline designed for materials science. It uses LangChain and LangGraph workflows to control the multimodal Qwen model, applying Pinecone metadata filters to accurately search, extract, and verify property data from research papers.

By processing both textual data and complex visual elements (such as charts, tables, and micrographs), MatScribe overcomes the limitations of standard text-only RAG systems, providing researchers with verified, highly accurate materials informatics.

## Key Features
* **Multimodal Data Ingestion:** Utilizes Vision-Language Models (Qwen2-VL) to parse and interpret text, tabular data, and scientific figures from unstructured PDFs.
* **Agentic Self-Correction Loop:** Implements a state machine consisting of a Retriever, Extractor, and Auditor. The Auditor independently verifies extracted claims against the source text to minimize AI hallucinations.
* **Semantic Search:** Leverages the LangChain-Pinecone integration to manage embeddings and execute high-precision similarity searches across the document knowledge base.
* **Hardware-Optimized UI:** Features a Streamlit dashboard with a conditional compute toggle, allowing users to bypass heavy image processing when only text extraction is required.

## System Architecture
The application follows a cyclic, agentic workflow:
1. **Ingestion:** Documents are parsed, embedded using `all-MiniLM-L6-v2`, and stored in Pinecone alongside structural metadata.
2. **Retrieval:** User queries are embedded, and relevant context is fetched using similarity search.
3. **Extraction:** The LLM analyzes the retrieved context to extract the specific materials property requested.
4. **Evaluation:** An Auditor agent reviews the extraction against the retrieved context. If the claim is unsupported, the workflow routes back to retry or flags the output as unverified.

## Technology Stack
* **Orchestration:** LangChain, LangGraph
* **Vision & Language Models:** Qwen2-VL-2B-Instruct
* **Embeddings:** HuggingFaceEmbeddings (all-MiniLM-L6-v2)
* **Vector Database:** Pinecone 
* **Document Processing:** pdfplumber, PIL
* **Frontend:** Streamlit
