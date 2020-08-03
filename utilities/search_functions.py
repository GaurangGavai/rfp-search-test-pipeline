import pandas as pd
import os
from utilities.helpfunctions import update_response_from_hit,sanitize_query, get_knn
import requests
import logging
from sentence_transformers import  SentenceTransformer
from flask import abort
import numpy as np
from elasticsearch import TransportError
from utilities.embedding_models import universal_embedding, init_bert_embeddings, query_embedding



ELASTICSEARCH_HOST = "localhost"
ELASTICSEARCH_PORT = "9200"



## Initialize the model
model = init_bert_embeddings()

logger = logging.getLogger("waitress")

def read_org(filename,org_name):
    org = pd.read_csv(filename)
    org = org.dropna()
    if org_name is not None:
        search_query = org['context'].tolist()
        selected_id = org['id'].tolist()
        return search_query, selected_id
    else:
        search_query = org['search_query'].tolist()
        index_selected = org['index_selected'].tolist()
        selected_id  = org['selected_id'].tolist()
        return search_query, index_selected, selected_id


def get_chunks_from_response(response_json: dict) -> list:

    """
    Get context/content chunks from elasticsearch response
    :param response_json: dict
    :return: list
    """

    matches = []

    for hit in response_json['hits']['hits']:

        ret = {}

        ret = update_response_from_hit(response=ret, hit=hit, prefix="_source", source_field="id", target_field="id")
        ret = update_response_from_hit(response=ret, hit=hit, prefix="_source", source_field="context",
                                       target_field="context")
        ret = update_response_from_hit(response=ret, hit=hit, prefix="_source", source_field="content",
                                       target_field="content")
        ret = update_response_from_hit(response=ret, hit=hit, prefix="_source", source_field="created_by",
                                       target_field="created_by")
        ret = update_response_from_hit(response=ret, hit=hit, prefix="_source", source_field="created_at",
                                       target_field="created_at")
        ret = update_response_from_hit(response=ret, hit=hit, prefix="_source", source_field="source",
                                       target_field="source")
        ret = update_response_from_hit(response=ret, hit=hit, prefix="_source", source_field="unwanted",
                                       target_field="unwanted")
        ret = update_response_from_hit(response=ret, hit=hit,prefix=None, source_field="_score",
                                       target_field="score")

        matches.append(ret)

    return matches




def get_all_results_from_elasticsearch(organization_index: str,
                                       query_body: dict,
                                       page_size: int,
                                       offset: int,
                                       es_host: str,
                                       es_port: int) -> dict:

    """
    Take a query and perform scrolling over the query results
    :param organization_index: str
    :param query_body: dict
    :param page_size: int
    :param offset: int
    :param es_host: str
    :param es_port: int
    :return: list
    """

    query_body["size"] = 10000

    # In elastic you have to perform first scroll and get the total count of documents and scroll id
    ret = requests.post(
        url="http://{0}:{1}/{2}/_search?scroll=1m".format(es_host,
                                                          es_port, organization_index),
        json=query_body)

    if ret.status_code != 200:
        message = "Problem with sending HTTP request to elasticsearch."
        logger.exception(message)
        abort(422, message)

    total_count = ret.json()["hits"]["total"]

    if page_size == -1:
        page_size = total_count

    es_res = {"matches": [],
              "total_count": total_count,
              "page_size": page_size,
              "offset": offset}

    matches = get_chunks_from_response(response_json=ret.json())

    # scroll_size = len(ret.json()['hits']['hits'])

    # Not all the wanted results were reached with the first search
    # if scroll_size < offset+page_size['value']:
    #
    #     logger.info("Have to scroll for more records.")
    #
    #     scroll_id = ret.json()["_scroll_id"]
    #     accumulated = scroll_size
    #
    #     body = {
    #         "scroll": "1m",
    #         "scroll_id": scroll_id
    #     }
    #
    #     while scroll_size > 0 and accumulated < offset+page_size:
    #
    #         ret = requests.post("http://{0}:{1}/_search/scroll".format(ELASTICSEARCH_HOST, ELASTICSEARCH_PORT),
    #                             json=body)
    #
    #         if ret.status_code != 200:
    #             message = "Problem when sending request to elasticsearch."
    #             logger.exception(message)
    #             abort(422, message)
    #
    #         matches += get_chunks_from_response(response_json=ret.json())
    #
    #         body["scroll_id"] = ret.json()["_scroll_id"]
    #         scroll_size = len(ret.json()['hits']['hits'])
    #         accumulated += scroll_size
    #
    # es_res["matches"] = matches[offset:offset+page_size]

    return matches



def es_search(organization_id, organization_index, req, db, include_unwanted=False):
    if not db.indices.exists(index=organization_index):
        message = "ES index not found for organizationId={0}".format(organization_id)
        abort(404, message)

    # query = req["query"]

    query = req

    body = {
        "query": {
            "bool": {
                "must": {}
            }
        }
    }

    sanitized_query = sanitize_query(query=query)

    body["query"]["bool"]["must"]["query_string"] = {
        "query": sanitized_query,
        "fields": ["content", "context"]
    }

    if not include_unwanted:
        body["query"]["bool"]["must_not"] = {"match": {"unwanted": 1}}

    logger.info("Performing elastic search.")

    if "page_size" in req and "offset" in req and req["page_size"] + req["offset"] <= 10000:

        logger.info("Executing search with page size and offset (<10000).")

        body["size"] = req["page_size"]
        body["from"] = req["offset"]

        ret = None
        try:
            ret = db.search(index=organization_index,  body=body)
        except TransportError:
            message = "Problem when sending request to elasticsearch."
            logger.exception(message)
            abort(422, message)

        es_res = {"matches": get_chunks_from_response(response_json=ret),
                  "total_count": ret["hits"]["total"],
                  "offset": req["offset"],
                  "page_size": body["size"]}

    else:

        logging.info("Executing search with scrolling to get all results.")

        page_size = -1
        offset = 0

        if "page_size" in req and req["page_size"]:
            page_size = req["page_size"]

        if "offset" in req:
            offset = req["offset"]

        es_res = get_all_results_from_elasticsearch(organization_index=organization_index,
                                                    query_body=body,
                                                    page_size=page_size,
                                                    offset=offset,
                                                    es_host=ELASTICSEARCH_HOST,
                                                    es_port=ELASTICSEARCH_PORT)

    logger.info("Done searching.")

    return es_res




def embed_query(query, embedding_type):
    if 'bert' in embedding_type:
        contents_embeddings = np.array(model.encode([query]))
    elif 'use' in embedding_type:
        contents_embeddings = query_embedding(query)

    return contents_embeddings




def embeddings_search(organization_index, req, embs_index,embedding_type):
    query = req
    page_size = -1

    logger.info("Performing embedding search.")

    if embedding_type:
       contents_embeddings = embed_query(query, embedding_type)


    emb_res = get_knn(query=contents_embeddings[0].tolist(),
                      index=organization_index,
                      k=max(page_size, 40),
                      embeddings_index=embs_index)

    return emb_res


def match_es_results(db_els, results, selected_idx,top_n):
    for idx, values in enumerate(results['matches']):
        if values['id_pair'] == selected_idx or idx < top_n:
            return idx + 1
        else:
            return -1



