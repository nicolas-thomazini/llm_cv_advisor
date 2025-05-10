# download minsearch.py
# if you download yet, you can comment this cell

# !wget https://raw.githubusercontent.com/alexeygrigorev/minsearch/main/minsearch.py

import minsearch

import json

with open('../data/documents.json', 'rt') as f_in:
    docs_raw = json.load(f_in)

documents = []

for course_dict in docs_raw:
    for doc in course_dict['documents']:
        doc['category'] = course_dict['category']
        documents.append(doc)

documents[0]

index = minsearch.Index(
    text_fields=["question", "text", "section"],
    keyword_fields=["category"]
)

question = 'Como posso melhorar meu LinkeDIN?'

index.fit(documents)

from openai import OpenAI

client = OpenAI(
    api_key="04K48DsYPmJyFaxtBOB5r2G9gsCJoMJo",
    base_url="https://api.mistral.ai/v1" 
)

response = client.chat.completions.create(
    model='mistral-medium',
    messages=[{"role": "user", "content": question}]
)

response.choices[0].message.content

def search(query):
    boost = {'questions': 3.0, 'section': 0.5}

    results = index.search(
        query=query,
        filter_dict={'category': 'Curriculo'},
        boost_dict=boost,
        num_results=10
    )

    return results

def build_prompt(query, search_results):
    prompt_template = """
        Você será um Chat Assistente. Responsa a QUESTION baseado no CONTEXT da base de dados FAQ.
        Use apenas os fatos do CONTEXT quando responder a QUESTION.

        QUESTION: {question}

        CONTEXT: {context}
""".strip()
    
    context = ""

    for doc in search_results:
        context = context + f"section: {doc['section']}\nquestion: {doc['question']}\nanswer: {doc['text']}"

    prompt = prompt_template.format(question=query, context=context).strip()
    return prompt

def llm(prompt):
    response = client.chat.completions.create(
        model='mistral-medium',
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

query = 'O que posso colocar ou usar no meu GitHub?'

def rag(query):
    search_results = search(query)
    prompt = build_prompt(query, search_results)
    answer = llm(prompt)
    return answer

rag(query)

rag('Posso colocar meus projetos pessoais no GitHub?')

from elasticsearch import Elasticsearch

es_client = Elasticsearch('http://localhost:9200')

index_settings = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0
    },
    "mappings": {
        "properties": {
            "text": {"type": "text"},
            "section": {"type": "text"},
            "question": {"type": "text"},
            "category": {"type": "keyword"} 
        }
    }
}

index_name = "category-question"

es_client.indices.create(index=index_name, body=index_settings)

documents[0]

from tqdm.auto import tqdm

for foc in tqdm(documents):
    es_client.index(index=index_name, document=doc)

query = 'Mesmo sem experiência profissional, posso colocar alguma experiência no meu Curriculo de algum projeto?'

def elastic_search(query):
    search_query = {
        "size": 5,
        "query": {
            "bool": {
                "must": {
                    "multi_match": {
                        "query": query,
                        "fields": ["question^7", "text", "section"],
                        "type": "best_fields"
                    }
                },
                "filter": {
                    "term": {
                        "category": "Curriculo"
                    }
                }
            }
        }
    }

    response = es_client.search(index=index_name, body=search_query)
    
    result_docs = []
    
    for hit in response['hits']['hits']:
        result_docs.append(hit['_source'])
    
    return result_docs

def rag(query):
    search_results = elastic_search(query)
    prompt = build_prompt(query, search_results)
    answer = llm(prompt)
    return answer

print(rag(query))
