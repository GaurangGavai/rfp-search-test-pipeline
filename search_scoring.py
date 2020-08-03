import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import logging
from utilities.embedding_functions import EmbeddingsIndex
from utilities.search_functions import embeddings_search, read_org
from utilities.search_ranking import search_index
import os
import argparse
import pandas as pd



parser = argparse.ArgumentParser(description='Data to create embeddings')

logger = logging.getLogger("waitress")

parser.add_argument('-s',  '--storage',          type=str,   help='location where the stored the embedding files')
parser.add_argument('-o',  '--organization_name',     type=str,   help='organization_name or the file name')
parser.add_argument('-e',  '--embedding_type',      type=str, choices= ['use-lite', 'bert'],  help='path to questions file (.tsv)')
parser.add_argument('-i',  '--input_file',      type=str,  help='path to the input file')
parser.add_argument('-ot',  '--output_file',      type=str,  help='path to the output file')


args = parser.parse_args()



STORAGE = args.storage
organization_name = args.organization_name
embedding_type = args.embedding_type
organization_index = args.organization_name


filename = args.input_file
organization_id = organization_name
org_name = organization_id



def search() :
    """
    Search embeddings index content.
    :return:
    """

    if org_name is not None:
        query_lst, selected_id = read_org(filename=filename, org_name=org_name)
    else:
        query_lst, selected_index, selected_id = read_org(filename=filename, org_name=None)


    top5_indx_emb, top3_indx_emb, top1_indx_emb = 0, 0, 0
    cnt_queries = 0

    for query,id in zip(query_lst, selected_id):

        operators = ["AND", "OR", "NOT", "\""]
        is_query_contains_operator = any(el in query for el in operators)

        is_elastic = is_query_contains_operator or len(query.split()) < 4

        if not is_elastic:
            cnt_queries += 1
            try:
                embs_index = EmbeddingsIndex(organization_dir=os.path.join(STORAGE, organization_id), load=True, embedding_type=embedding_type,
                                             logger=logger)
                embs = embeddings_search(organization_index, query, embs_index, embedding_type=embedding_type)
                embs_index.unload()
                top5_indx_emb, top3_indx_emb, top1_indx_emb = search_index(embs, top5_indx_emb, top3_indx_emb, top1_indx_emb, id)


            except Exception as e:
                print(e)
                pass
    return top5_indx_emb, top3_indx_emb, top1_indx_emb,cnt_queries

def write_to_csv():
    top5_indx_emb, top3_indx_emb, top1_indx_emb,cnt_queries = search()
    final_dict = dict()
    final_dict.update({'organization_index': organization_index})
    final_dict.update({'embedding_type': embedding_type})
    final_dict.update({'filename_path': filename})
    final_dict.update({'hits@5': top5_indx_emb})
    final_dict.update({'hits@3': top3_indx_emb})
    final_dict.update({'hits@1': top1_indx_emb})
    final_dict.update({'Total_Queries': cnt_queries})
    final_df = pd.DataFrame(final_dict.items())
    final_df.columns = ['Keys', 'Values']
    final_df.to_csv(args.output_file, index=False)



if __name__ == "__main__":
    write_to_csv()

