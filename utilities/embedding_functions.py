from annoy import AnnoyIndex
import numpy
import os
import h5py
from threading import Thread
# from docunderstanding.proposalgeneration.exceptions import EmbeddingsDbNotSynced

"""
Embeddings index represents an approach for search over closest embeddings.
Annoy index is used for computing the approximate nearest neighbors using the Locality Sensitive Hashing.
The graph has to be rebuilt every time something is added or removed.
"""


class EmbeddingsIndex:

    def __init__(self, organization_dir, logger, embedding_type, load=False):

        """
        :param organization_dir: str - path to resources
        :param logger: logging object
        :param load: bool - True if used for generating content
        """

        self.organization_dir = organization_dir
        self.logger = logger
        self.embedding_type = embedding_type
        if 'use' in self.embedding_type:
            self.f_dim = 1024
        elif 'bert' in self.embedding_type:
            self.f_dim = 1536


        if load:

            self.logger.debug("Loading model.")
            self.t = AnnoyIndex(self.f_dim)
            self.t.load(os.path.join(self.organization_dir, "lshforest_" + self.embedding_type + ".ann"))

            self.logger.debug("Loading ids.")
            with h5py.File(os.path.join(self.organization_dir, "data_" + self.embedding_type + ".hdf5"), "r") as f:
                self.ids = numpy.array(f["id"])

            if len(self.ids) != self.t.get_n_items():
                raise EmbeddingsDbNotSynced

    def build_index(self) -> None:

        """
        Load embeddings and build a new index. Run it as a deamon because it might take more time with
        huge content databases.
        :return: None
        """

        self.logger.debug("Starting recommendation model building daemon.")
        self.retrieve_and_build()
        # t = Thread(target=self.retrieve_and_build, daemon=True)
        # t.start()

    def build_empty_index(self) -> None:

        """
        Build empty index - used when building index after deleting takes time.
        :return: None
        """

        t = AnnoyIndex(self.f_dim)

        trees_num = 10  # Number of trees in the index
        t.build(trees_num)
        t.save(os.path.join(self.organization_dir, "lshforest_" + self.embedding_type + ".ann"))
        print('organization ann has been created')
        t.unload()

    def retrieve_and_build(self):

        """
        Build empty index - used when building index after deleting takes time.
        :return: None
        """

        t = AnnoyIndex(self.f_dim)

        trees_num = 10  # Number of trees in the index
        t.build(trees_num)

        t.save(os.path.join(self.organization_dir, "lshforest.ann"))
        t.unload()

    def retrieve_and_build(self):

        """
        Retrieve embeddings stored in HDF5 files and build and save the graph.
        :return: None
        """

        try:
            embs = self.retrieve_embeddings()

            self.logger.debug("Creating annoy index. Embeddings count: %s.", len(embs))
            t = AnnoyIndex(self.f_dim)
            for i in range(0, len(embs)):
                t.add_item(i, embs[i])

            trees_num = 10  # Number of trees in the index
            t.build(trees_num)

            self.logger.debug("Building annoy index.")

            t.save(os.path.join(self.organization_dir, "lshforest_" + self.embedding_type + ".ann"))

            self.logger.debug("Saved annoy index.")
            t.unload()

        except Exception:
            message = "Failed to build recommendation model."
            self.logger.exception(message)

    def get_knn(self, vector, k=3):

        """
        Get k closest neighbors to a vector
        :param vector: embedding of a query
        :param k: parameter specifying how many neighbors we want
        :return: dict - dict with results
        """

        self.logger.debug("Getting nearest neighbors.")
        # Get closest neighbors
        ind, dist = self.t.get_nns_by_vector(vector, n=k, include_distances=True)

        ret = {"hits": []}

        if len(self.ids) != self.t.get_n_items():
            raise EmbeddingsDbNotSynced

        for d, i in zip(dist, ind):
            ret["hits"].append({"id": self.ids[i], "score": d})

        self.logger.debug("Nearest neighbors found.")
        return ret

    def retrieve_embeddings(self) -> numpy.ndarray:

        """
        Retrieve embeddings from a pandas dataframe
        :return: numpy.ndarray
        """

        self.logger.debug("Retrieving embeddings.")
        with h5py.File(os.path.join(self.organization_dir, "data_" + self.embedding_type + ".hdf5"), "r") as f:

            contexts = numpy.array(f["context"])
            contents = numpy.array(f["content"])

        ret = numpy.hstack((contexts, contents))

        self.logger.debug("Retrieved embeddings.")

        return ret

    def unload(self) -> None:

        """
        Unload index from memory
        :return: None
        """

        self.t.unload()
        del self.t



class EmbeddingsDbNotSynced(Exception):

    def __init__(self):

        super().__init__("Embeddings model and dataset no synchronized.")