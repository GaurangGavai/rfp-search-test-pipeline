import tensorflow as tf
import tensorflow.compat.v1 as tfv1
import tensorflow_hub as hub
from utilities.helpfunctions import process_in_sparse_format, prepare_qa_pairs_for_indexing, process_to_IDs_in_sparse_format
from sentence_transformers import  SentenceTransformer
import numpy as np

tf.compat.v1.disable_eager_execution()
import sentencepiece as spm


## Developing Universal sentence embeddings


def init_embeddings_model():

    """
    Initialize model for embedding pieces of content into a vector
    :return: None
    """
    g = tfv1.Graph()
    with g.as_default():
        model_path = "https://tfhub.dev/google/universal-sentence-encoder-lite/2"
        module = hub.Module(model_path)
        sts_input1_ = tfv1.sparse_placeholder(tfv1.int64, shape=[None, None])
        embeddings_ = module(inputs=dict(
                values=sts_input1_.values,
                indices=sts_input1_.indices,
                dense_shape=sts_input1_.dense_shape))

    return g, sts_input1_, embeddings_, module


def init_bert_embeddings():
    """
    Initialize BERT-BASE-NLI-MEAN model
    :return:
    """

    model = SentenceTransformer('bert-base-nli-mean-tokens')
    print('BERT model has been loaded')
    return model



model = init_bert_embeddings()

graph, sts_input1, embeddings, module = init_embeddings_model()


with tfv1.Session(graph=graph) as sess:
    spm_path = sess.run(module(signature="spm_path"))

sp = spm.SentencePieceProcessor()
sp.Load(spm_path)



def universal_embedding(qa_pairs, added_idxs):


    with tfv1.Session(graph=graph) as sess:
        spm_path = sess.run(module(signature="spm_path"))

    sp = spm.SentencePieceProcessor()
    sp.Load(spm_path)



    contexts = None
    contents = None
    org_contexts = None
    str_contents = None
    matching_ids = None
    try:

        matching_ids, str_contents, contents, org_contexts, contexts = prepare_qa_pairs_for_indexing(qa_pairs=qa_pairs)

    except KeyError:
        message = "There is no 'content' in a question answer pair."
        print(message)



    contexts_embeddings = None
    contents_embeddings = None

    added_idxs = None

    tf.compat.v1.reset_default_graph()

    try:

        values_cntx, indices_cntx, dense_shape_cntx = process_in_sparse_format(sp, contexts, added_idxs)

        values_cntn, indices_cntn, dense_shape_cntn = process_in_sparse_format(sp, contents, added_idxs)

        with tfv1.Session(graph=graph) as session:

            session.run([tfv1.global_variables_initializer(), tfv1.tables_initializer()])

            # Embed contexts and contents
            contexts_embeddings = session.run([embeddings],
                                              feed_dict={sts_input1.values: values_cntx,
                                                         sts_input1.indices: indices_cntx,
                                                         sts_input1.dense_shape: dense_shape_cntx})[0]
            contents_embeddings = session.run([embeddings],
                                              feed_dict={sts_input1.values: values_cntn,
                                                         sts_input1.indices: indices_cntn,
                                                         sts_input1.dense_shape: dense_shape_cntn})[0]

        return contexts_embeddings, contents_embeddings, matching_ids

    except RuntimeError:
        message = "Tensorflow runtime error."
        print(message)
        return None, None


def bert_embedding(qa_pairs):

    matching_ids, str_contents, contents, org_contexts, contexts = prepare_qa_pairs_for_indexing(qa_pairs=qa_pairs)

    context_embeddings = np.array(model.encode(contexts))

    content_embeddings = np.array(model.encode(contexts))

    print('BERT embeddings have been created')

    return context_embeddings, content_embeddings,matching_ids


def query_embedding(query):


    # with tfv1.Session(graph=graph) as sess:
    #     spm_path = sess.run(module(signature="spm_path"))
    #
    # sp = spm.SentencePieceProcessor()
    # sp.Load(spm_path)


    qa_pair = {"content": query, "context": query}
    cleaned_query_df = prepare_qa_pairs_for_indexing([qa_pair])
    # cleaned_query = cleaned_query_df["content_for_embedding"][0]

    values, indices, dense_shape = process_to_IDs_in_sparse_format(sp, query)

    with tfv1.Session(graph=graph) as session:

        session.run([tfv1.global_variables_initializer(), tfv1.tables_initializer()])

        query_embeddings = session.run(
            embeddings,
            feed_dict={sts_input1.values: values,
                       sts_input1.indices: indices,
                       sts_input1.indices: indices,
                       sts_input1.dense_shape: dense_shape})


    return query_embeddings



