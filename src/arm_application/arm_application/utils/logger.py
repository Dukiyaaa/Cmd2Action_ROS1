# utils/logger.py
import logging
import os
from datetime import datetime
from typing import Optional, Dict, Any


class ProjectLogger:
    def __init__(self, module_name: str, log_dir: str = "logs", level: int = logging.DEBUG):
        self.module_name = module_name
        self.log_dir = log_dir
        self.level = level

        os.makedirs(self.log_dir, exist_ok=True)

        self.logger = logging.getLogger(module_name)
        self.logger.setLevel(level)
        self.logger.propagate = False

        if not self.logger.handlers:
            formatter = logging.Formatter(
                fmt="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )

            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)
            console_handler.setFormatter(formatter)

            log_filename = datetime.now().strftime("run_%Y%m%d.log")
            file_handler = logging.FileHandler(
                os.path.join(self.log_dir, log_filename),
                encoding="utf-8"
            )
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)

            self.logger.addHandler(console_handler)
            self.logger.addHandler(file_handler)

    def _format_message(
        self,
        func: str,
        msg: str,
        task_id: Optional[str] = None,
        code: Optional[str] = None,
        **kwargs: Any
    ) -> str:
        parts = [f"[{func}]"]

        if task_id is not None:
            parts.append(f"task_id={task_id}")

        if code is not None:
            parts.append(f"code={code}")

        parts.append(msg)

        if kwargs:
            detail = " ".join(f"{k}={v}" for k, v in kwargs.items())
            parts.append(f"| {detail}")

        return " ".join(parts)

    def debug(self, func: str, msg: str, task_id: Optional[str] = None, code: Optional[str] = None, **kwargs: Any):
        self.logger.debug(self._format_message(func, msg, task_id, code, **kwargs))

    def info(self, func: str, msg: str, task_id: Optional[str] = None, code: Optional[str] = None, **kwargs: Any):
        self.logger.info(self._format_message(func, msg, task_id, code, **kwargs))

    def warning(self, func: str, msg: str, task_id: Optional[str] = None, code: Optional[str] = None, **kwargs: Any):
        self.logger.warning(self._format_message(func, msg, task_id, code, **kwargs))

    def error(self, func: str, msg: str, task_id: Optional[str] = None, code: Optional[str] = None, **kwargs: Any):
        self.logger.error(self._format_message(func, msg, task_id, code, **kwargs))


def get_logger(module_name: str, log_dir: str = "logs") -> ProjectLogger:
    return ProjectLogger(module_name=module_name, log_dir=log_dir)