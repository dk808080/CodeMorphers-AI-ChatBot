# CodeMorphers-AI-ChatBot
High Level: Create a Conversational AI bot for answering user support queries

### Hosted URL - 
https://ff6d729d1cd0f4d4c9.gradio.live/

### PPT and Demo - 
https://drive.google.com/drive/folders/1h0RYuOLU9bcLB0b5JRwBeBw2Zr6QQpKw?usp=drive_link

#### Request to the jury - ** please refer the PPT from above drive link, as direct upload for PPT resulted into a broken file.

System to fetch documentation from a website, process and embed the content for semantic search, and finally, retrieve relevant information in response to a user's query. The user interacts with the system through a simple Gradio-based interface. Here's a breakdown of each component:

## System workflow 
![image](https://github.com/user-attachments/assets/e04ee3c3-c6f6-4428-a07c-1daf862d01d2)


## Libraries and Modules Setup

The code starts by installing some key Python libraries using pip install for packages like qdrant-haystack, fastembed, groq, and gradio.
Key imports include:
requests: For making HTTP requests (used to fetch the sitemap and documentation).
BeautifulSoup: For web scraping and parsing HTML data.
nltk: For tokenizing text (breaking it into sentences).
fastembed: For embedding text into vector representations.
QdrantDocumentStore & QdrantEmbeddingRetriever: For storing and retrieving documents in an embedding-based search system.
groq: For performing inference using the Groq AI model.

## Functions for Sitemap and Data Extraction
These functions handle retrieving and processing the sitemap of a given documentation website:

get_sitemap_data(url):

Takes a base URL and appends /sitemap.xml to retrieve the sitemap, which contains the list of pages available in the documentation.
Uses requests.get to fetch the sitemap XML, and handles potential HTTP errors.
extract_urls_from_sitemap(sitemap_data):

Parses the fetched XML sitemap using BeautifulSoup to extract all URLs tagged within <url><loc>.
Returns a list of documentation URLs.
fetch_and_store_documentation(base_url):

Fetches the sitemap and loops over the URLs to download the HTML content of each page.
Filters out irrelevant HTML tags (<script>, <style>, <nav>, etc.).
Stores the plain text from the HTML content in a dictionary where the keys are the URLs and the values are the page content.


## Text Processing
The downloaded documentation content is processed into tokenized sentences:

fetch_and_process(base_url):
Calls fetch_and_store_documentation() to retrieve and clean the documentation data.
For each URL, it tokenizes the content into sentences using nltk.sent_tokenize.
The result is a dictionary where each URL maps to a list of sentences.


## Embedding the Documentation with FastEmbed
The processed sentences are transformed into embeddings (vector representations) using the FastEmbed model:

embed_documents(documents):
Initializes the FastEmbed model.
Iterates through the tokenized sentences and generates embeddings for each using embedding_model.embed().
The sentence-embedding pairs are stored for each URL.

## Storing and Retrieving Documents in Qdrant
The code then uses Qdrant to store and retrieve documents based on their embeddings:

QdrantDocumentStore:

This is a vector storage database where you store the documents along with their embeddings.
The documents are written to the Qdrant in-memory store with document_store.write_documents(ingestion_data).
QdrantEmbeddingRetriever:

This component retrieves documents from the Qdrant store based on how similar their embeddings are to a query embedding.


## Inference using Groq AI
The core inference process involves querying the stored documentation:

groqInference(query, top_k=20):
Embeds the user's query using FastEmbed.
Retrieves the top relevant documents from Qdrant based on their similarity to the query embedding.
Forms a prompt using the retrieved content and sends it to the Groq model for answering the user’s query.
The Groq model processes the prompt and returns a generated response based on the provided documentation content.

## Gradio Interface
Finally, a Gradio interface is set up to interact with the system through a web app:

iface = gr.Interface():
This sets up a simple Gradio web interface where users can enter their query in a text box.
When they submit a question, it calls the groqInference() function to generate a response.
The response is displayed in a textbox as the output.

## Launching the Interface
iface.launch(): This launches the Gradio interface locally, allowing users to input queries and receive responses.
Summary:
The system fetches documentation pages, cleans the content, and embeds it for semantic search.
It stores the embeddings using Qdrant and retrieves relevant content based on user queries.
Finally, the Groq AI model is used to answer user questions using the retrieved content, and this interaction is facilitated through a simple Gradio interface.
