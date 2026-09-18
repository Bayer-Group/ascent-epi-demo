# coding=utf-8
__author__ = "Angelo Ziletti"
__maintainer__ = "Angelo Ziletti"
__date__ = "22/02/24"

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class Table1:
    def __init__(self, input_question: Optional[str] = None):
        self.input_question = input_question
        self.input_text_cohort_generator = None

        self.cohort_creation_question = None
        self.cohort_df = {}

        # derived quantities
        self.rwd_request = None

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove the non-pickleable 'assistant' attribute from the state
        if "assistant" in state:
            del state["assistant"]
        return state
