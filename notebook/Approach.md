# SHL Assessment Recommendation System - My Approach

To tackle the SHL Assessment Recommendation challenge, I began by thoroughly cleaning and preprocessing the assessment catalog data, which included standardizing formats and extracting useful features like numerical duration values. My initial retrieval approach was purely lexical, using BM25. While it provided a basic baseline, the Recall@10 and MAP@10 were quite low—under 10%—which signaled a need for more contextual understanding of queries.

I then integrated semantic search using Sentence-BERT (SBERT) and indexed the assessment embeddings using FAISS. This gave a noticeable boost in performance, especially for more abstract or nuanced job descriptions. However, the results were still not optimal in many cases. Interestingly, for some technical queries, BM25 proved more useful than semantic search. So, I experimented with Reciprocal Rank Fusion (RRF) to combine BM25 and SBERT rankings, which helped balance lexical precision with semantic breadth.

To further improve query understanding, I added Gemini API integration to extract structured intent and keywords from natural language job descriptions. Combining Gemini-enhanced queries with BM25 and SBERT outputs resulted in better matches, improving the system’s accuracy substantially.

I continued iterating by introducing filtering based on duration and test type metadata to discard assessments irrelevant by constraints. 

I also tested a two-stage hybrid pipeline where BM25 fetched candidates and SBERT re-ranked them semantically, which pushed the performance further.

Ultimately, these iterative enhancements—semantic embeddings, LLM-assisted parsing, metadata filtering, hybrid ranking—culminated in a model achieving **30% Recall@10** and **21% MAP@10** on the official test set. While there’s still room for improvement through supervised fine-tuning or LLM-based reranking, I’ve built a zero-shot, generalizable, and interpretable solution that performs competitively without any training data and minimal latency.
