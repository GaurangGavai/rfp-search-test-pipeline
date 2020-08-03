#!/bin/bash

storage=$1 #path to store the embeddings and the index
organization_name=$2 ## Organization name or test data name
embedding_type=$3 ## Embedding type can be USE-LITE or BERT
input_file=$4 ## file containing the queries
output_file=$5 ##output location of the file



python3 insert_test_set.py --storage=$storage --organization_name=$organization_name --embedding_type=$embedding_type --input_file=$input_file

python3 search_scoring.py --storage=$storage --organization_name=$organization_name --embedding_type=$embedding_type --input_file=$input_file --output_file=$output_file