import logging
from tqdm import tqdm

class TqdmHandler(logging.StreamHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            self.handleError(record)

class TaskLoggerAdapter(logging.LoggerAdapter):
    def __init__(self, logger):
        super().__init__(logger, {})
        self.default_level = 0
        self.disable_pbar = False

    def process(self, msg, kwargs):
        level = kwargs.pop("depth", self.default_level)
        indent = '  ' * level
        return f"{indent}{msg}", kwargs

    def indent(self):
        class Indenter:
            def __init__(self, adapter):
                self.adapter = adapter
            def __enter__(self):
                self.adapter.default_level += 1
            def __exit__(self, exc_type, exc_val, exc_tb):
                self.adapter.default_level -= 1
        return Indenter(self)

_logger_adapter = None

def set_logger(logger_adapter: TaskLoggerAdapter) -> None:
    global _logger_adapter
    _logger_adapter = logger_adapter

def get_logger() -> TaskLoggerAdapter:
    if _logger_adapter is None:
        raise RuntimeError("Logger has not been initialized.")
    return _logger_adapter