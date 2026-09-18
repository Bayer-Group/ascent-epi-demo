import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from ascent_domain.omop.data.queries import querylib_loader
from ascent_domain.omop.data.queries.query_library import QueryLibrary
from ascent_platform.llm.factory import create_assistant

logger = logging.getLogger(__name__)


class QueryLibraryManager:
    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self.querylib = None
        self.logger = logging.getLogger(self.__class__.__name__)

    @classmethod
    def get_instance(cls):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = cls()
        return cls._instance

    def load_querylib(self, main_path: Optional[str] = None, querylib_file: Optional[str] = None, embedding_model: str = "BAAI/bge-large-en-v1.5"):
        if self.querylib is not None:
            return self.querylib

        main_path = main_path or os.getcwd()
        querylib_file = querylib_file or querylib_loader.get_latest_querylib_file(main_path)

        querylib = QueryLibrary(
            querylib_name="patient_counts",
            source="gold_label_dec_2023",
            querylib_source_file=None,
            col_question="QUESTION",
            col_question_masked="QUESTION_MASKED",
            col_query_w_placeholders="QUERY_SNOWFLAKE_WITH_PLACEHOLDERS",
            col_query_executable="QUERY_SNOWFLAKE_RUNNABLE",
        )
        querylib = querylib.load(querylib_file=querylib_file)
        querylib.load_embedding_model(embedding_model_name=embedding_model)

        self.logger.info(f"Embedding loaded from {querylib_file}")
        self.querylib = querylib
        return self.querylib


@dataclass
class RAGConfig:
    main_path: Optional[Path] = None
    log_folder: Optional[Path] = None
    querylib_file: Optional[Path] = None
    assistant_type: str = "gpt"
    model_name: Optional[str] = None
    top_k_prompt: int = 5
    top_k_screening: int = 50
    sim_threshold: float = 0.0
    prompt_template: str = None
    database: Optional[str] = None
    additional_params: Dict[str, Any] = field(default_factory=dict)


class AssistantState:
    """
    Represents the state of an assistant, including the main assistant
    and optionally assistant_answers.
    """

    def __init__(self, assistant, assistant_answers=None):
        self.assistant = assistant
        self.assistant_answers = assistant_answers


class RAGProcessor:
    def __init__(self, config: RAGConfig):
        self.config = config
        self.querylib_manager = QueryLibraryManager.get_instance()

        # Default state for when no specific database is provided
        self.default_state = self._create_state()

        # Dictionary to store states for different databases
        self.database_states = {}

    @property
    def default_assistant(self):
        return self.default_state.assistant

    @property
    def default_assistant_answers(self):
        return self.default_state.assistant_answers

    @property
    def querylib(self):
        """The query library, loaded on first access.

        This read the manager's attribute directly, which is only populated
        once somebody has called load_querylib(). The QA path does that
        explicitly (qa.py), the cohort path does not -- so cohort generation
        got None and failed several frames later with "'NoneType' object has
        no attribute 'get_masked_question'", naming neither the library nor
        the load.

        load_querylib() returns early when already loaded, so making the
        property self-sufficient costs nothing and removes the ordering
        dependency between callers.
        """
        return self.querylib_manager.load_querylib()

    def _create_state(self) -> AssistantState:
        """
        Create an AssistantState object based on the current configuration.
        """
        if self.config.assistant_type == "huggingface_custom_model":
            if not self.config.additional_params.get("model_path") or not self.config.additional_params.get("run_name"):
                raise ValueError("model_path and run_name must be provided for huggingface_custom assistant")
            assistant = create_assistant(
                assistant_type=self.config.assistant_type,
                model_path=self.config.additional_params.get("model_path"),
                run_name=self.config.additional_params.get("run_name"),
            )
            return AssistantState(assistant)
        else:
            assistant = create_assistant(
                assistant_type=self.config.assistant_type,
                model_name=self.config.model_name,
            )
            assistant_answers = create_assistant(
                assistant_type=self.config.assistant_type,
                model_name=self.config.model_name,
            )
            return AssistantState(assistant, assistant_answers)

    def get_state(self) -> Dict:
        """
        Get the current state of the RAGProcessor.
        Returns a dictionary containing serializable state data.
        """

        # Serialize the default state
        return {
            "default_state": self.default_state,
        }
