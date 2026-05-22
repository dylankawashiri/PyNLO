from __future__ import annotations

from enum import Enum
import logging
import os
import yaml

logger = logging.getLogger(__name__)

class Collection(Enum):
    GENERAL_FIBERS = "general_fibers"

class Fibers(Enum):
    SIMPLE_FIBER = "simple_fiber"

class FiberLoader:
    def __init__(self, *, fiber_collection: Collection = Collection.GENERAL_FIBERS, file_dir: str | None = None):
        if file_dir is None:
            root = os.path.abspath(os.path.dirname(__file__))
        else:
            root = file_dir
        with open(f"{root}/{fiber_collection.value}.yaml", "r") as f:
            self.config = yaml.safe_load(f)
        logger.info(f"Successfully loaded from {root}/{fiber_collection.value}.yaml.")
        self.fibers = [fiber for fiber in self.config]

    def print_fiber_list(self):
        for i, fiber in enumerate(self.fibers):
            logger.info(f"Fiber #{i}: {fiber}")
    
    def get_fiber(self, name: str) -> dict[str, str | list[int] | float]:
        return self.config[name]
