!pip install qdrant-haystack
!pip install fastembed
!pip install groq
!pip install gradio

import os
import numpy as np
import time
import requests
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize
from fastembed import TextEmbedding
from groq import Groq
import gradio as gr

from haystack.dataclasses.document import Document
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever

import nltk
nltk.download('punkt')


import requests
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize

def get_sitemap_data(url):
    """
    Retrieves the sitemap.xml data from the given URL.

    Args:
        url (str): The base URL of the documentation website.

    Returns:
        str: The content of the sitemap.xml file.
    """
    sitemap_url = f"{url}/sitemap.xml"
    try:
        response = requests.get(sitemap_url)
        response.raise_for_status()  # Raise an exception for bad status codes
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching sitemap: {e}")
        return None

def extract_urls_from_sitemap(sitemap_data):
    """
    Extracts URLs from the given sitemap.xml data.

    Args:
        sitemap_data (str): The content of the sitemap.xml file.

    Returns:
        list: A list of URLs extracted from the sitemap.
    """
    soup = BeautifulSoup(sitemap_data, 'xml')
    urls = []
    for url_tag in soup.find_all('url'):
        loc_tag = url_tag.find('loc')
        if loc_tag:
            urls.append(loc_tag.text)
    return urls

def fetch_and_store_documentation(base_url):
    """
    Fetches documentation content from URLs and stores them in a dictionary.

    Args:
        base_url (str): The base URL of the documentation website.

    Returns:
        dict: A dictionary where keys are URLs and values are filtered HTML content.
    """
    sitemap_data = get_sitemap_data(base_url)
    if sitemap_data:
        urls = extract_urls_from_sitemap(sitemap_data)
        # Limit to the first 100 URLs
        urls = urls[:100]
        
        docs = {}  # Initialize an empty dictionary

        for url in urls:
            try:
                response = requests.get(url)
                response.raise_for_status()

                soup = BeautifulSoup(response.text, 'html.parser')
                # Filter out unwanted tags using BeautifulSoup (adjust as needed)
                for tag in ['script', 'style', 'nav', 'aside', 'footer']:
                    for element in soup.find_all(tag):
                        element.decompose()

                docs[url] = soup.get_text(separator=' ')  # Store filtered HTML content
                print(f"Fetched and stored content from: {url}")

            except requests.exceptions.RequestException as e:
                print(f"Error fetching {url}: {e}")

        return docs
    else:
        return None
    
# Sentence Tokenization
def fetch_and_process(base_url):
    documentation_data = fetch_and_store_documentation(base_url) 
    if documentation_data:
        for url, content in documentation_data.items():
            sentences = sent_tokenize(content)
            documentation_data[url] = sentences
        return documentation_data
    return None


base_url = "https://help.gohighlevel.com/support/"
documentation_data = fetch_and_process(base_url)


from fastembed import TextEmbedding
# Initialize the TextEmbedding model
embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5", cache_dir="./embeddings")

def embed_documents(documents):
    for url, sentences in documentation_data.items():
        try:
            embeddings = []
            for sentence in sentences:
                # Embed document using FastEmbed
                embedding = np.array(list((embedding_model.embed([sentence]))))
                
                # Append the embedding to the list of embeddings
                embeddings.append((sentence,embedding))
            
            documentation_data[url] = embeddings
        except:
            pass
    return documentation_data

# Perform embedding generation
documentation_data = embed_documents(documentation_data)

ingestion_data = []

document_store = QdrantDocumentStore(
    ":memory:",
    index="Document",
    embedding_dim=384,
    recreate_index=True,
    hnsw_config={"m": 16, "ef_construct": 64}  # Optional
)

for url, sentences in documentation_data.items():
    ingestion_data.append(Document(content=sentences[0][0], embedding=sentences[0][1][0], meta={"url": url}))


document_store.write_documents(ingestion_data)
retriever = QdrantEmbeddingRetriever(document_store=document_store)


def groqInference(query, top_k=20):
    query_embedding = list((embedding_model.embed([query])))
    retrieved_content = retriever.run(list(query_embedding[0]), top_k=top_k)
    
    client = Groq(
        api_key="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",
    )

    prompt = f"""Below is given a Documentation and answer the question asked in the end:
    {retrieved_content['documents'][0].content}
    \n\n\n
    {query}
    """

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="llama3-8b-8192",
    )

    return chat_completion.choices[0].message.content

response = groqInference(query, 10)

iface = gr.Interface(
    fn=groqInference,
    inputs=[
        gr.Textbox(label="Query", placeholder="Enter your question here")
    ],
    outputs=[gr.Textbox(label="Generated Response")],
    title="QnA Bot",
    description="Enter thequestion to get a generated response based on the retrieved text from the Documentation."
)

iface.launch()
