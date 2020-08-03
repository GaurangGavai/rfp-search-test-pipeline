from bs4 import BeautifulSoup
import re
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import TransportError
from utilities.embedding_functions import EmbeddingsIndex

def get_es_setup() -> dict:

    """
    Setting up similarity search for elasticsearch
    :return: dict
    """

    settings = {

        "settings": {
            "index": {
                "similarity": {
                    "my_similarity": {
                        "type": "DFR",
                        "basic_model": "g",
                        "after_effect": "l",
                        "normalization": "h2",
                        "normalization.h2.c": "3.0"
                    }
                }
            },
            "analysis": {
                "filter": {
                    "english_stemmer": {
                        "type": "stemmer",
                        "language": "english"
                    }
                },
                "analyzer": {
                    "search_analyzer": {
                        "type": "custom",
                        "tokenizer": "standard",
                        "filter": [
                            "lowercase",
                            "english_stemmer"
                        ],
                        "char_filter": [
                            "html_strip"
                        ]
                    }
                }
            }
        },
        "mappings": {

                "properties": {
                    "content": {
                        "type": "text",
                        "similarity": "my_similarity",
                        "analyzer": "search_analyzer",
                        "fields": {
                            "raw": {
                                "type": "text"
                            }
                        }
                    },
                    "context": {
                        "type": "text",
                        "similarity": "my_similarity",
                        "analyzer": "search_analyzer",
                        "fields": {
                            "raw": {
                                "type": "text"
                            }
                        }
                    },
                    "id": {
                        "type": "keyword"
                    }
                }

        }

    }

    return settings

def get_content_until_max_length(content: str, max_length: int) -> str:

    """
    Get content until specified max_length. Do not cut it in the middle of a word - find the next space.
    :param content: str - text
    :param max_length: int - maximum expected length
    :return: str
    """

    if max_length == 0:
        return ""

    if content[max_length:].find(" ") != -1:
        return content[:max_length + content[max_length:].find(" ")]
    else:
        return content[:max_length]



def get_text_from_semistructured_content(semi_str_content):

    """
    Return the text content of a semi-structured element. BeautifulSoup checks the validity of content and tries
    to fix major issues with unclosed elements etc.
    :param semi_str_content: str
    :return: str
    """

    return BeautifulSoup(semi_str_content, "html.parser").getText()




def prepare_qa_pairs_for_indexing(qa_pairs: list) -> tuple:

    """
    List of qa pairs where every qa_pair should have a context, content and an id
    :param qa_pairs: list
    :return: tuple
    """

    matching_ids = []  # Id of a QA pair
    str_contents = []  # Original content with structure HTML formatting
    contents = []  # Summary of text taken from HTML enriched chunks for embedding

    org_contexts = []  # Clean version of the context
    contexts = []  # Summary of the context text

    for qa in qa_pairs:

        if is_empty_qa(qa=qa):
            raise KeyError()


        if len(qa["content"].strip()) == 0:
            continue


        if "id" in qa:
            matching_ids.append(qa["id"])
        else:
            matching_ids.append(None)

        content = qa["content"]
        str_contents.append(content)

        # Assuming 2000 could be the max lengths and would contain all relevant information
        # Main reason is for decreasing memory and computation time.
        contents.append(get_content_until_max_length(content=get_text_from_semistructured_content(content),
                                                     max_length=2000))

        if "context" not in qa or qa["context"] is None or len(qa["context"].strip()) == 0:
            qa["context"] = ""

            # Use the content if the context is not there.
            # Embeddings search takes the similarity of [query][query] vs. [context][content] embeddings
            # [query][query] vs. [None][content] would make it difficult to retrieve the answer from ndim space.
            # It could very well match a [None][other_content] and that is a False Positive
            # [query][query] vs. [part_of_content][content] makes it easier to retrieve.

            cleaned_context = qa["content"]
            org_contexts.append("")
        else:
            # Questions are usually short and main part is at the beginning. Clean out numberings at the beginning.
            cleaned_context = clean_content(qa["context"])
            org_contexts.append(cleaned_context)

        # Main reason is for decreasing memory and computation time.
        contexts.append(get_content_until_max_length(content=cleaned_context,
                                                     max_length=400))

    return matching_ids, str_contents, contents, org_contexts, contexts

def is_empty_qa(qa: dict) -> bool:

    """
    Check whether the QA pair is empty == there is no content in the qa_pairs
    :param qa: dict - should contain "content" and "context" field.
    :return: bool
    """

    no_content = False
    if "content" not in qa or qa["content"] is None:
        no_content = True

    return no_content



def clean_content(content: str) -> str:

    """
    Clean ol and ul listing characters from a string
    :param content: str
    :return: str
    """

    content = re.sub("^\s*([a-z]\.|\d+\.|[VIXDCLM]+\.)+\w+\s", "", string=content, flags=re.IGNORECASE)
    content = re.sub("^\s*([a-z]\.|\d+\.|[VIXDCLM]+\.)+\s", "", string=content, flags=re.IGNORECASE)
    content = re.sub("^\s*\t*((\d+|[a-z]{1}|M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3}))(\.|:)|\(?(\d+|[a-z]{1}|M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3}))\)\s)", "", string=content, flags=re.IGNORECASE)
    content = re.sub("^\d+\s+", "", string=content, flags=re.IGNORECASE)
    content = re.sub("^\s*[^0-9a-z\(\)\s\t\n\"]\s*", "", string=content, flags=re.IGNORECASE)
    content = re.sub("^\s*o\s+", "", string=content, flags=re.IGNORECASE)
    content = re.sub("\s*\n+\s*", "\n", string=content)

    return content.strip()


def process_to_IDs_in_sparse_format(sp, sentences):
    if isinstance(sentences, list):
        sent = [get_content_until_max_length(clean_content(task["content"]), max_length=400) for task in
                    sentences]
    elif isinstance(sentences, str):
        sent = [get_content_until_max_length(clean_content(sentences), max_length=400)]
    ids = [sp.EncodeAsIds(x) for x in sent]
    max_len = max(len(x) for x in ids)
    dense_shape = (len(ids), max_len)
    values = [item for sublist in ids for item in sublist]
    indices = [[row, col] for row in range(len(ids)) for col in range(len(ids[row]))]
    return (values, indices, dense_shape)



def process_in_sparse_format(sp, sentences, added_idx=None):
    if added_idx is not None:
        ids = [sp.EncodeAsIds(x) for idx, x in enumerate(sentences) if idx in added_idx]
    else:
        ids = [sp.EncodeAsIds(x) for idx, x in enumerate(sentences)]
    max_len = max(len(x) for x in ids)
    dense_shape = (len(ids), max_len)
    values = [item for sublist in ids for item in sublist]
    indices = [[row, col] for row in range(len(ids)) for col in range(len(ids[row]))]
    return (values, indices, dense_shape)

def process_to_IDs_in_sparse_format(sp, sentences):
    if isinstance(sentences, list):
        sent = [get_content_until_max_length(clean_content(task["content"]), max_length=400) for task in
                    sentences]
    elif isinstance(sentences, str):
        sent = [get_content_until_max_length(clean_content(sentences), max_length=400)]
    ids = [sp.EncodeAsIds(x) for x in sent]
    max_len = max(len(x) for x in ids)
    dense_shape = (len(ids), max_len)
    values = [item for sublist in ids for item in sublist]
    indices = [[row, col] for row in range(len(ids)) for col in range(len(ids[row]))]
    return (values, indices, dense_shape)





def get_knn(query: list, index: str,
            k: int, embeddings_index: EmbeddingsIndex) -> list:
    """
    Get best search results from embeddings space
    :param es: Elasticsearch instance
    :param query: list - embedd query point
    :param index: organization_id - index to elasticsearch
    :param doc_type: document type in elasticsearch
    :param k: k nearest neighbors
    :param embeddings_index: EmbeddingsIndex instance representing index of embeddings for knn search
    :return: list - list of best matches from elastic search
    """

    merged_embedding = query + query

    # Get closest neighbors
    ret = embeddings_index.get_knn(merged_embedding, k=k)
    return ret["hits"]



def update_response_from_hit(response: dict, hit: dict, prefix, source_field: str, target_field: str) -> dict:

    """
    Take a hit from elasticsearch and lookup the desired field. Copy it to response if found.
    :param response: dict
    :param hit: dict
    :param prefix: str/None - string name of the parent field in the hierarchy
    :param source_field: str
    :param target_field: str
    :return: dict
    """

    if prefix is None:
        response[target_field] = hit[source_field]

    else:

        if prefix in hit and source_field in hit[prefix]:

            response[target_field] = hit[prefix][source_field]

    return response



def sanitize_query(query: str) -> str:

    """
    Sanitize query so that I can query elasticsearch (escape special characters except '"', exclude '<', '>' from query)
    :param query: str
    :return: str
    """
    special_symbols = {'\\', '+', '-', '!', '(', ')', ':', '^', '[', ']', '{', '}', '~', '*', '?', '|', '&', '/'}

    sanitized_query = ''
    for character in query:
        if character in ['<','>']:
            continue
        if character in special_symbols:
            sanitized_query += '\\%s' % character
        else:
            sanitized_query += character

    return sanitized_query