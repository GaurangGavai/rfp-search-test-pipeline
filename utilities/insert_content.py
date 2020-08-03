import h5py
import os


"""
Functions for storing the vector representation of the data
"""


def create_new_embeddings_dataset(organization_dir: str, ids: list, contexts: list, contents: list, embedding_type:str) -> None:

    """
    Create new dataset if it doesnt exist yet. Store it as HFD5 file. Together with it, save the lookup json
    file that converts the string id to row index. Lookup should be constant with hashing function.
    :param organization_dir: str - path to the resources dir
    :param ids: list of str - list if unique ids for each content.
    :param contexts: list of numpy arrays - embeddings of contexts
    :param contents: list of numpy arrays - embeddings of contents
    :return: None
    """

    ids = [x.encode("utf-8") for x in ids]

    with h5py.File(os.path.join(organization_dir, "data_" + embedding_type + ".hdf5"), "w") as f:
        dt = h5py.special_dtype(vlen=str)
        f.create_dataset("id",
                         dtype=dt,
                         shape=(len(ids),),
                         data=ids,
                         chunks=True,
                         maxshape=(None,))
        f.create_dataset("context",
                         shape=contexts.shape,
                         dtype=contexts.dtype,
                         data=contexts,
                         chunks=True,
                         maxshape=(None, None,))
        f.create_dataset("content",
                         shape=contents.shape,
                         dtype=contents.dtype,
                         data=contents,
                         chunks=True,
                         maxshape=(None, None,))


def append_to_embeddings_dataset(organization_dir: str, ids: list, contexts: list, contents: list, embedding_type :str) -> None:

    """
    Append new data to existing files. Load the JSON file and extend it with new values. Append to HDF5 file.
    :param organization_dir: str - path to the resources dir
    :param ids: list of str - list if unique ids for each content.
    :param contexts: list of numpy arrays - embeddings of contexts
    :param contents: list of numpy arrays - embeddings of contents
    :return: None
    """

    ids = [x.encode("utf-8") for x in ids]

    with h5py.File(os.path.join(organization_dir, "data_" + embedding_type + " .hdf5"), "a") as f:
        f["id"].resize(f["id"].shape[0] + len(ids), axis=0)
        f["id"][-len(ids):] = ids

        f["context"].resize(f["context"].shape[0] + len(contexts), axis=0)
        f["context"][-len(contexts):] = contexts

        f["content"].resize(f["content"].shape[0] + len(contents), axis=0)
        f["content"][-len(contents):] = contents
