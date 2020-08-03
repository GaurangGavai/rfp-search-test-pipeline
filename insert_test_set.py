# from elasticsearch import  Elasticsearch
# from sentence_transformers import  SentenceTransformer
# from utils.helpfunctions import prepare_qa_pairs_for_indexing
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from utilities.embedding_functions import EmbeddingsIndex
from utilities.insert_content import create_new_embeddings_dataset, append_to_embeddings_dataset
from utilities.embedding_models import universal_embedding, bert_embedding
import os
import logging
import argparse
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger("waitress")

parser = argparse.ArgumentParser(description='Data to create embeddings')

# Dataset params
parser.add_argument('-s',  '--storage',          type=str,   help='location to store the embedding files')
parser.add_argument('-o',  '--organization_name',     type=str,   help='organization_name or the file name')
parser.add_argument('-e',  '--embedding_type',      type=str,  help='path to questions file (.tsv)')
parser.add_argument('-i',  '--input_file',      type=str,  help='path to the input file')

args = parser.parse_args()


STORAGE = args.storage
organization_name = args.organization_name
embedding_type = args.embedding_type
organization_index = args.organization_name


qa_pairs = list()
list_values = list()
input_q_a = pd.read_csv(args.input_file)
input_q_a.dropna(inplace=True)
context = input_q_a['context'].tolist()
content = input_q_a['context'].tolist()
idx = input_q_a['id'].tolist() if 'id' in input_q_a.columns else list(range(0, len(content)))

list_values.append(context)
list_values.append(content)
list_values.append(idx)

added_ids = list()
added_idxs = list()


for logit_pairs in tqdm(zip(*list_values)):
    id = logit_pairs[2]
    content = logit_pairs[1]
    context = logit_pairs[0]

    body = {
        "context": context,
        "content": content,
        "id": id
    }

    qa_pairs.append(body)
    added_ids.append(id)

added_idxs = None





contexts_embeddings, contents_embeddings, matching_ids = universal_embedding(qa_pairs, added_idxs)

##Bert Sentence embeddings

context_embeddings_bert, content_embeddings_bert, matching_ids_bert = bert_embedding(qa_pairs)




if not os.path.exists(os.path.join(STORAGE, organization_name)):
    os.mkdir(os.path.join(STORAGE, organization_name))
    print('Directory created - {}'.format(os.path.join(STORAGE, organization_name)))



# for embedding_type in embedding_config:

if 'use' in embedding_type:

    if not os.path.exists(os.path.join(STORAGE, organization_name, "data_" + embedding_type + '.hdf5')):



        try:
            create_new_embeddings_dataset(organization_dir=os.path.join(STORAGE, organization_name),
                                          contexts=contexts_embeddings,
                                          contents=contents_embeddings,
                                          embedding_type = embedding_type,
                                          ids=matching_ids)
            embs_index = EmbeddingsIndex(organization_dir=os.path.join(STORAGE, organization_name),
                                         embedding_type=embedding_type, logger=logger)
            embs_index.build_index()
            print('Annoy index successfully built')

        except OSError:
            message = "Resource directory not found."
            print(message)


    else:
        print('Embeddings already exist, try deleting and rerunning again,proceeding to search')



elif 'bert' in embedding_type:
    if not os.path.exists(os.path.join(STORAGE, organization_name, "data_" + embedding_type + '.hdf5')):



        try:
            create_new_embeddings_dataset(organization_dir=os.path.join(STORAGE, organization_name),
                                          contexts=context_embeddings_bert,
                                          contents=content_embeddings_bert,
                                          embedding_type = embedding_type,
                                          ids=matching_ids_bert)
            embs_index = EmbeddingsIndex(organization_dir=os.path.join(STORAGE, organization_name),
                                         embedding_type=embedding_type, logger=logger)
            embs_index.build_index()
            print('Annoy index successfully built')
        except OSError:
            message = "Resource directory not found."
            print(message)


    else:
        print('Embeddings already exist, try deleting and rerunning again, procceeding to search')

        # append_to_embeddings_dataset(organization_dir=os.path.join(STORAGE, organization_name),
        #                              contexts=context_embeddings_bert,
        #                              contents=content_embeddings_bert,
        #                              embedding_type = embedding_type,
        #                              ids=matching_ids_bert)



