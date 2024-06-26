## Problem Statement

Designed a solution utilizing open-source large language models to efficiently respond to queries where the answers need to be derived from extensive PDF documents (exceeding 100 pages).

Example of queries that should be answered:

- Explain the theme of the movie?
- Who are the characters?
- How many male and female characters are in the movie?
- Does the script pass the Bechdel test?
- What is the role of Deckard in the movie?

## My Approach
In order to solve the problem I first used *pdfplumber* library to read the input pdf and convert its data to string format to fed into the LLM (NousResearch/Llama-2-7b-hf). In order to make the whole process faster what I did was instead of simply using the whole document together, I split them into paragraphs and then I used sentence embeddings in order to do perform semantic search based on the query. This allowed me to quickly answer the questions.

Here is the demo link:

https://github.com/shalinimeena/QnA-system-using-LLMs/assets/101329715/3648802c-b087-416d-b5b0-05a13be373e7

deploy link:
https://qna-system-using-llms-isbzqx6ndr6pzdoxy5a2fl.streamlit.app/

