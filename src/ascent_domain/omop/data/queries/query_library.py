# coding=utf-8
__author__ = "Angelo Ziletti"
__maintainer__ = "Angelo Ziletti"
__date__ = "24/11/23"

import asyncio
import logging
import os.path
import pickle
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from functools import cache
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from ascent_domain.omop.models.inference.prompts import entity_masking

tqdm.pandas()

logger = logging.getLogger(__name__)


@cache
def _import_sklearn_preprocessing():
    import sklearn.preprocessing as pp

    return pp


def normalize(*args, **kwargs):
    """sklearn.preprocessing.normalize with import caching"""
    pp = _import_sklearn_preprocessing()
    return pp.normalize(*args, **kwargs)


@cache
def _import_sentence_transformer() -> type:
    from sentence_transformers import SentenceTransformer as cls

    return cls


def make_sentence_transformer(*args, **kwargs):
    """sentence_transformers.SentenceTransformer with import caching"""
    cls = _import_sentence_transformer()
    return cls(*args, **kwargs)


class QueryLibrary:
    """Collection of queries for retrieval augmented generation"""

    def __init__(
        self,
        querylib_name: str,
        source: str,
        querylib_source_file: object,
        col_question: str,
        col_question_masked: str,
        col_query_w_placeholders: str,
        col_query_executable: Optional[str] = None,
        date_live: Optional[date] = None,
        storage_type: str = "pickle",  # Add storage_type parameter
    ) -> None:
        self.querylib_name = querylib_name
        self.date_live = date_live
        self.source = source
        self.col_question = col_question
        self.col_question_masked = col_question_masked
        self.col_query_w_placeholders = col_query_w_placeholders
        self.col_query_executable = col_query_executable
        self.storage_type = storage_type

        if querylib_source_file:
            df_querylib = pd.read_excel(querylib_source_file)
            self.df_querylib = df_querylib
        else:
            self.df_querylib = pd.DataFrame()
        self.label_encoder = None
        self.label_encoder_dict = None

        self.embeddings = []

        self.embedding_model = None

    @staticmethod
    def _init_sqlite_db(db_path: str) -> None:
        """Initialize SQLite database schema"""
        with sqlite3.connect(db_path) as conn:
            # Drop existing tables if they exist
            conn.execute("DROP TABLE IF EXISTS queries")
            conn.execute("DROP TABLE IF EXISTS embeddings")
            conn.execute("DROP TABLE IF EXISTS metadata")
            conn.execute("DROP TABLE IF EXISTS label_encoder")

            # Create queries table
            conn.execute("""
                CREATE TABLE queries (
                    id INTEGER PRIMARY KEY,
                    question TEXT NOT NULL,
                    question_masked TEXT,
                    query_with_placeholders TEXT,
                    query_executable TEXT,
                    question_type TEXT
                )
            """)

            # Create embeddings table with matrix_shape column
            conn.execute("""
                CREATE TABLE embeddings (
                    id INTEGER PRIMARY KEY,
                    model_name TEXT,
                    embed_matrix BLOB,
                    matrix_shape TEXT
                )
            """)

            # Create metadata table
            conn.execute("""
                CREATE TABLE metadata (
                    key TEXT PRIMARY KEY,
                    value BLOB
                )
            """)

            # Create table for label encoder if needed
            conn.execute("""
                CREATE TABLE label_encoder (
                    id INTEGER PRIMARY KEY,
                    class_name TEXT,
                    encoded_value INTEGER
                )
            """)

            # Add indices
            conn.execute("CREATE INDEX idx_question_type ON queries(question_type)")
            conn.execute("CREATE INDEX idx_model_name ON embeddings(model_name)")

            conn.commit()

    def __len__(self):
        return len(self.df_querylib)

    def save(self, querylib_file: str) -> None:
        """Save the query library to either pickle or SQLite"""
        """Save the query library to either pickle or SQLite"""
        if not self.verify_embeddings():
            raise ValueError("Embedding verification failed")

        if self.storage_type == "pickle" or querylib_file.endswith(".pkl"):
            with open(querylib_file, "wb") as out_file:
                pickle.dump(self, out_file)
            logger.info(f"Saved query library to pickle file: {querylib_file}")
        elif self.storage_type == "sqlite" or querylib_file.endswith(".db"):
            self._init_sqlite_db(querylib_file)
            with sqlite3.connect(querylib_file) as conn:
                # Save metadata about the QueryLibrary instance
                metadata = {
                    "querylib_name": self.querylib_name,
                    "source": self.source,
                    "col_question": self.col_question,
                    "col_question_masked": self.col_question_masked,
                    "col_query_w_placeholders": self.col_query_w_placeholders,
                    "col_query_executable": self.col_query_executable,
                    "date_live": self.date_live.isoformat() if self.date_live else None,
                    "label_encoder": sqlite3.Binary(pickle.dumps(self.label_encoder)) if self.label_encoder else None,
                    "label_encoder_dict": sqlite3.Binary(pickle.dumps(self.label_encoder_dict)) if self.label_encoder_dict else None,
                    "embedding_model": sqlite3.Binary(pickle.dumps(self.embedding_model)) if self.embedding_model else None,
                }

                # Save metadata
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS metadata (
                        key TEXT PRIMARY KEY,
                        value BLOB
                    )
                """)
                for key, value in metadata.items():
                    conn.execute("INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)", (key, value))

                # Save queries
                self.df_querylib.to_sql("queries", conn, if_exists="replace", index=True)

                # Drop existing embeddings table if it exists
                conn.execute("DROP TABLE IF EXISTS embeddings")

                # Create embeddings table with correct schema
                conn.execute("""
                    CREATE TABLE embeddings (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        model_name TEXT,
                        embed_matrix BLOB,
                        matrix_shape TEXT
                    )
                """)

                # Save embeddings with shape information
                if self.embeddings:
                    for embedding in self.embeddings:
                        embed_matrix = embedding["embed_matrix"]
                        conn.execute(
                            """
                            INSERT INTO embeddings 
                            (model_name, embed_matrix, matrix_shape) 
                            VALUES (?, ?, ?)
                            """,
                            (
                                embedding["model_name"],
                                sqlite3.Binary(embed_matrix.tobytes()),
                                str(embed_matrix.shape),  # Save shape as string
                            ),
                        )
                conn.commit()
                logger.info(f"Saved query library to SQLite database: {querylib_file}")

    def load_embedding_model(self, embedding_model_name):
        if embedding_model_name is None:
            error_msg = (
                f"\n{'=' * 80}\n"
                f"❌ EMBEDDING MODEL NAME IS NONE!\n"
                f"{'=' * 80}\n"
                f"   Cannot load embedding model with None name.\n"
                f"\n"
                f"   Please ensure you pass a valid model name, e.g.:\n"
                f"   - 'BAAI/bge-large-en-v1.5' (HuggingFace model)\n"
                f"   - A local path to a downloaded model\n"
                f"{'=' * 80}\n"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        try:
            logger.info(f"Loading embedding model: {embedding_model_name}")
            self.embedding_model = make_sentence_transformer(embedding_model_name)
            logger.info(f"Successfully loaded embedding model: {embedding_model_name}")
        except Exception as e:
            error_msg = (
                f"\n{'=' * 80}\n"
                f"❌ FAILED TO LOAD EMBEDDING MODEL!\n"
                f"{'=' * 80}\n"
                f"   Model name: {embedding_model_name}\n"
                f"   Error: {e}\n"
                f"\n"
                f"   Possible causes:\n"
                f"   1. No internet connection (model needs to be downloaded)\n"
                f"   2. Corporate proxy blocking HuggingFace downloads\n"
                f"   3. Model not found on HuggingFace Hub\n"
                f"   4. Insufficient disk space for model cache\n"
                f"\n"
                f"   Solutions:\n"
                f"   - Check your internet/proxy settings\n"
                f"   - Try downloading the model manually first\n"
                f"   - Use a locally cached model path\n"
                f"{'=' * 80}\n"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def verify_embeddings(self):
        """Verify the integrity of embeddings before saving"""
        if not self.embeddings:
            logger.warning("No embeddings found to verify")
            return False

        for idx, embedding in enumerate(self.embeddings):
            if "model_name" not in embedding:
                logger.error(f"Embedding {idx} missing model_name")
                return False
            if "embed_matrix" not in embedding:
                logger.error(f"Embedding {idx} missing embed_matrix")
                return False

            matrix = embedding["embed_matrix"]
            if not isinstance(matrix, np.ndarray):
                logger.error(f"Embedding {idx} matrix is not a numpy array")
                return False

            logger.info(f"Embedding {idx} verified: {matrix.shape}")
        return True

    @staticmethod
    def load(querylib_file: Union[str, Path]):
        """Load the query library from either pickle or SQLite"""
        logger.info(f"Loading query library from: {querylib_file}")

        # if the input is a string, convert it to path to correctly detect the storage format
        querylib_file = Path(querylib_file) if isinstance(querylib_file, str) else querylib_file

        # Check if file exists FIRST - provide clear error message
        if not querylib_file.exists():
            error_msg = (
                f"\n{'=' * 80}\n"
                f"❌ QUERYLIB FILE NOT FOUND!\n"
                f"{'=' * 80}\n"
                f"   File path: {querylib_file}\n"
                f"\n"
                f"   Please check:\n"
                f"   1. The file path is correct\n"
                f"   2. The file exists at the specified location\n"
                f"   3. There are no typos in the filename\n"
                f"      (e.g., 'querylib_2025116.db' vs 'querylib_20251116.db')\n"
                f"{'=' * 80}\n"
            )
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        if querylib_file.suffix.lower() == ".pkl":
            try:
                with open(querylib_file, "rb") as in_file:
                    query_lib = pickle.load(in_file)
                    if hasattr(query_lib, "embeddings"):
                        if not query_lib.embeddings:
                            logger.warning("Loaded query library has empty embeddings list")
                        else:
                            for idx, embed_dict in enumerate(query_lib.embeddings):
                                embed_shape = embed_dict["embed_matrix"].shape if "embed_matrix" in embed_dict else "No matrix"
                                logger.info(f"Embedding {idx} shape: {embed_shape}")
                    else:
                        logger.warning("Loaded query library has no embeddings attribute")

                    logger.info("Query library loaded from pickle file")
                return query_lib
            except Exception as e:
                error_msg = (
                    f"\n{'=' * 80}\n"
                    f"❌ QUERYLIB PICKLE FILE FAILED TO LOAD!\n"
                    f"{'=' * 80}\n"
                    f"   File path: {querylib_file}\n"
                    f"   Error: {e}\n"
                    f"\n"
                    f"   The pickle file exists but could not be loaded.\n"
                    f"   Possible reasons:\n"
                    f"   1. The file is corrupted\n"
                    f"   2. The file was created with an incompatible Python version\n"
                    f"   3. Required classes are not available in the current environment\n"
                    f"\n"
                    f"   Try using a .db file instead or regenerate the pickle file.\n"
                    f"{'=' * 80}\n"
                )
                logger.error(error_msg)
                raise ValueError(error_msg) from e
        elif querylib_file.suffix.lower() == ".db":
            try:
                with sqlite3.connect(str(querylib_file)) as conn:
                    # Load metadata
                    metadata = dict(conn.execute("SELECT key, value FROM metadata").fetchall())

                    # Create new instance with metadata
                    query_lib = QueryLibrary(
                        querylib_name=metadata["querylib_name"],
                        source=metadata["source"],
                        querylib_source_file=None,  # We're loading from DB
                        col_question=metadata["col_question"],
                        col_question_masked=metadata["col_question_masked"],
                        col_query_w_placeholders=metadata["col_query_w_placeholders"],
                        col_query_executable=metadata.get("col_query_executable"),
                        date_live=datetime.fromisoformat(metadata["date_live"].decode()) if metadata["date_live"] else None,
                        storage_type="sqlite",
                    )

                    # Load binary objects directly from the database
                    if metadata["label_encoder"]:
                        query_lib.label_encoder = pickle.loads(metadata["label_encoder"])
                    if metadata["label_encoder_dict"]:
                        query_lib.label_encoder_dict = pickle.loads(metadata["label_encoder_dict"])
                    if metadata["embedding_model"]:
                        query_lib.embedding_model = pickle.loads(metadata["embedding_model"])

                    # Load queries
                    query_lib.df_querylib = pd.read_sql("SELECT * FROM queries", conn)

                    # Load embeddings
                    query_lib.embeddings = []
                    for model_name, embed_matrix, matrix_shape in conn.execute(
                        "SELECT model_name, embed_matrix, matrix_shape FROM embeddings"
                    ).fetchall():
                        # Convert string shape back to tuple
                        shape = tuple(map(int, matrix_shape.strip("()").split(",")))
                        try:
                            # matrix = np.frombuffer(embed_matrix).reshape(shape)
                            matrix = np.frombuffer(embed_matrix, dtype=np.float32).reshape(shape)
                            query_lib.embeddings.append({"model_name": model_name, "embed_matrix": matrix})
                            logger.debug(f"Loaded embedding matrix with shape: {shape}")
                        except ValueError as e:
                            logger.error(f"Error reshaping matrix: {e}")
                            logger.error(f"Matrix size: {len(embed_matrix)}, Attempted shape: {shape}")
                            raise

                    logger.info("Query library loaded from SQLite database")
                    return query_lib

            except sqlite3.OperationalError as e:
                error_msg = (
                    f"\n{'=' * 80}\n"
                    f"❌ QUERYLIB DATABASE STRUCTURE INVALID!\n"
                    f"{'=' * 80}\n"
                    f"   File path: {querylib_file}\n"
                    f"   SQLite error: {e}\n"
                    f"\n"
                    f"   The file exists but is not a valid querylib database.\n"
                    f"   Possible reasons:\n"
                    f"   1. The database is missing required tables ('metadata', 'queries', 'embeddings')\n"
                    f"   2. The file is not a SQLite database\n"
                    f"   3. The database was created with a different schema version\n"
                    f"   4. The file is corrupted\n"
                    f"\n"
                    f"   Try using a different querylib file.\n"
                    f"{'=' * 80}\n"
                )
                logger.error(error_msg)
                raise ValueError(error_msg) from e

            except Exception as e:
                error_msg = (
                    f"\n{'=' * 80}\n"
                    f"❌ QUERYLIB FAILED TO LOAD!\n"
                    f"{'=' * 80}\n"
                    f"   File path: {querylib_file}\n"
                    f"   Error: {e}\n"
                    f"\n"
                    f"   The file exists but could not be loaded.\n"
                    f"   Try using a different querylib file or check the error above.\n"
                    f"{'=' * 80}\n"
                )
                logger.error(error_msg)
                raise ValueError(error_msg) from e

        else:
            raise ValueError(f"Unsupported file type: {querylib_file}")

    def extract_idx_records(self, values_to_extract, source_col):
        idx_records = self.df_querylib.index[self.df_querylib[source_col].isin(values_to_extract)].tolist()
        return idx_records

    def extract_embed_matrix(self, value_rows_to_extract, extract_col_name, embedding):
        # Find the rows in the ontology that match name_rows_to_extract
        idx_records = self.extract_idx_records(value_rows_to_extract, extract_col_name)

        # Get the corresponding embedding matrix
        embed_matrix = embedding["embed_matrix"][idx_records]

        # Get the names from the matrix so they match the embeddings
        value_rows_embed = self.df_querylib.loc[idx_records][extract_col_name].reset_index(drop=True)

        return embed_matrix, value_rows_embed

    @staticmethod
    def add_separator_to_input_entities(lst, sep="[SEP_P]"):
        joined_list = []
        for inner_list in lst:
            joined_list.append(f" {sep} ".join(inner_list))
        return joined_list

    def get_similar_questions(
        self,
        samples,
        top_k=5,
        sim_threshold=0.95,
        normalize_score=True,
        col_search=None,
        max_rows=1000,
        tmp_dir=None,
        export_txt=False,
        question_type=None,
    ):
        if col_search is None:
            col_search = self.col_question

        df_querylib_selected = self.df_querylib

        # Apply question type filter if specified
        if question_type is not None:
            if question_type not in ["QA", "COHORT_GENERATOR"]:
                raise ValueError("question_type must be either 'QA' or 'COHORT_GENERATOR'")
            df_querylib_selected = df_querylib_selected[df_querylib_selected["QUESTION_TYPE"] == question_type]

        embed_matrix, names_avail = self.extract_embed_matrix(
            value_rows_to_extract=df_querylib_selected[self.col_question].tolist(),
            extract_col_name=self.col_question,
            embedding=self.embeddings[0],
        )

        # Cast the input samples in a dataframe for convenience
        samples_with_sep = self.add_separator_to_input_entities(samples)
        df_input_names = pd.DataFrame(samples_with_sep, columns=[self.col_question])

        df_input_names_list = [df_input_names[i : i + max_rows] for i in range(0, df_input_names.shape[0], max_rows)]

        df_recap_recs_list = []
        df_recs_list = []

        outfile_recap_recs_list = []
        outfile_recs_list = []

        # Parallel processing for get_similar_names
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(
                    self.get_similar_names,
                    df_input_chunk,
                    names_avail,
                    embed_matrix=embed_matrix,
                    embedding_model=self.embedding_model,
                    col_search=col_search,
                    normalize_score=normalize_score,
                    top_k=top_k,
                    sim_threshold=sim_threshold,
                    export_txt=export_txt,
                ): idx
                for idx, df_input_chunk in enumerate(df_input_names_list)
            }

            for future in as_completed(futures):
                idx = futures[future]
                df_recap_recs, df_recs = future.result()

                if tmp_dir is not None:
                    outfile_recap_recs = os.path.join(tmp_dir, f"recap_recs_{idx}")
                    outfile_recs = os.path.join(tmp_dir, f"recs_{idx}")

                    with open(outfile_recap_recs, "wb") as out_file:
                        pickle.dump(df_recap_recs, out_file)

                    with open(outfile_recs, "wb") as out_file:
                        pickle.dump(df_recs, out_file)

                    outfile_recap_recs_list.append(outfile_recap_recs)
                    outfile_recs_list.append(outfile_recs)

                    del df_recap_recs, df_recs
                else:
                    df_recap_recs_list.append(df_recap_recs)
                    df_recs_list.extend(df_recs)

        if tmp_dir is not None:
            df_recap_recs_list = []
            for outfile_recap in outfile_recap_recs_list:
                with open(outfile_recap, "rb") as out_file:
                    df_recap_recs_list.append(pickle.load(out_file))

            df_recs_list = []
            for outfile_rec in outfile_recs_list:
                with open(outfile_rec, "rb") as out_file:
                    df_recs_list.extend(pickle.load(out_file))

        df_recap_recs = pd.concat(df_recap_recs_list)

        return df_recap_recs, df_recs_list

    def get_similar_names(
        self,
        df,
        classes,
        embed_matrix,
        embedding_model,
        col_search,
        suffix="unsupervised",
        normalize_score=True,
        top_k=20,
        sim_threshold=0.0,
        top_k_limit=None,
        export_txt=True,
    ):

        if top_k_limit is None:
            top_k_limit = len(classes)

        logger.debug("Retrieving the most similar classes")

        # remove leading and trailing spaces
        df[col_search] = df[col_search].astype(str).str.strip()

        # Compute embeddings in batches (assuming embedding_model can handle batch input)
        text_embeddings = embedding_model.encode(df[self.col_question].tolist(), normalize_embeddings=True)

        logger.debug(f"Input embed_matrix shape: {embed_matrix.shape}")
        logger.debug(f"Text embeddings shape: {text_embeddings.shape}")

        logger.debug(f"Text embedding size: {text_embeddings.nbytes / 10**6} (Mb)")
        logger.debug(f"text_embedding shape: {text_embeddings.shape}")

        if normalize_score:
            text_embeddings = normalize(text_embeddings)
            embed_matrix = normalize(embed_matrix)

        # Efficient matrix multiplication
        sim_matrix = text_embeddings @ embed_matrix.T

        logger.debug(f"Similarity matrix size: {sim_matrix.nbytes / 10**6} (Mb)")
        logger.debug(f"sim_matrix shape: {sim_matrix.shape}")

        # Efficient top-k selection
        idx_match_sorted = np.argpartition(-sim_matrix, kth=top_k_limit - 1, axis=1)[:, :top_k_limit]

        matched_classes_list = []
        scores_list = []
        code_list = []
        df_class_score_list = []

        for idx_input in range(sim_matrix.shape[0]):
            idx_match_input = idx_match_sorted[idx_input]
            sim_matrix_row_sorted = -np.partition(-sim_matrix[idx_input], kth=top_k_limit - 1)[:top_k_limit]
            class_text_sorted = classes[idx_match_input]

            df_class_scores = pd.DataFrame(
                zip(class_text_sorted, sim_matrix_row_sorted),
                columns=["Class", "Score"],
            )
            df_class_scores = df_class_scores.nlargest(top_k, "Score")

            matched_classes_list.append(df_class_scores["Class"].tolist())
            scores_list.append(df_class_scores["Score"].tolist())

            df_class_scores[self.col_question] = df_class_scores["Class"]
            code_list.append(df_class_scores[self.col_question].tolist())

            if not export_txt:
                df_class_scores = df_class_scores.drop(["Class"], axis=1, errors="ignore")

            df_class_score_list.append(df_class_scores)

        df[f"rec_{suffix}_questions"] = code_list
        df[f"rec_{suffix}_scores"] = scores_list

        return df, df_class_score_list

    def get_df_recs(self, question, top_k, sim_threshold, question_type):
        df_recap_recs, df_recs_list = self.get_similar_questions(question, top_k=top_k, sim_threshold=sim_threshold, question_type=question_type)

        # here the list is only one element long because we pass only one question
        df_recs_list_merged = []
        for df_recs in df_recs_list:
            df_merged = df_recs.merge(self.df_querylib, on=self.col_question, how="left")
            df_recs_list_merged.append(df_merged)

        # here we only have one question thus the list is always one element
        # TO DO: this is not ideal and it should be improve later
        df_recs_list_out = df_recs_list_merged[0]
        return df_recs_list_out

    async def text_sql_template_for_rag(
        self,
        question_masked,
        top_k_screening,
        top_k_prompt,
        sim_threshold,
        reverse_order=False,
        rag_random=False,  # Parameter for random retrieval
        drop_first=False,  # Parameter to drop the first element
        question_type=None,
    ):
        # df_recs_list_out = self.get_df_recs([[question_masked]], top_k=top_k_screening, sim_threshold=sim_threshold, question_type=question_type)

        df_recs_list_out = await asyncio.to_thread(
            self.get_df_recs, [[question_masked]], top_k=top_k_screening, sim_threshold=sim_threshold, question_type=question_type
        )

        # If reverse_order is True, reverse the order of the DataFrame
        if reverse_order:
            df_recs_list_out = df_recs_list_out.sort_index(ascending=False)

        # If drop_first is True, drop the first element from the DataFrame
        if drop_first:
            logger.warning("Dropping first element of the retrieved queries")
            df_recs_list_out = df_recs_list_out.drop(df_recs_list_out.index[0])

        # If rag_random is True, randomly select one sample from the top-k
        if rag_random:
            logger.warning("Using random retrieval for RAG")
            df_recs_list_out = df_recs_list_out.sample(n=top_k_screening)

        # Keep only the top_k_prompt elements
        df_recs_list_out = df_recs_list_out.head(top_k_prompt)

        initial_sentence = "\n\nYou might find these example queries helpful: "

        # Modify the template based on question_type
        if question_type == "COHORT_GENERATOR":
            text_sql_template = (
                initial_sentence
                + "\n\n"
                + "\n\n".join(
                    f"#Example inclusion/exclusion criteria:\n{rec[self.col_question]}\n#Example SQL query:\n{rec[self.col_query_w_placeholders]}"
                    for rec in df_recs_list_out.to_dict("records")
                )
            )
        else:  # None or "QA"
            text_sql_template = (
                initial_sentence
                + "\n\n"
                + "\n\n".join(
                    f"#Example Question:\n{rec[self.col_question]}\n#Example SQL query:\n{rec[self.col_query_w_placeholders]}"
                    for rec in df_recs_list_out.to_dict("records")
                )
            )

        return text_sql_template, df_recs_list_out

    @staticmethod
    async def get_masked_question(question, assistant, reset_conversation=True, mask="DRUG_CLASS"):
        """
        :param prompts: List of prompts
        :param question: User question
        :param assistant: Assistant to use
        :param reset_conversation: True or False to reset the conversation
        :param mask: Mask to apply
        :return: masked question, question
        """
        prompt = entity_masking.format(question=question)

        assistant.add_message(role="user", message=prompt)
        masked_question = await assistant.get_response()

        if mask in masked_question:
            if masked_question.count(mask) > 1 or (masked_question.count(mask) == 1 and masked_question.count("DRUG") > 1):
                question += " Can you output also intermediate results for each drug class?"

        logger.info(f"Masked question: {masked_question}")

        if reset_conversation:
            assistant.reset_conversation()

        return masked_question, question
