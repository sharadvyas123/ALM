"""tqdm display selection for notebooks with or without ipywidgets."""
from __future__ import annotations

from tqdm import tqdm as TerminalProgress


class NotebookTextProgress(TerminalProgress):
    """Update one notebook text display when widget support is unavailable.

    Retains tqdm's counters, ETA, rate, and postfix formatting without sending
    carriage-return redraws to the notebook's captured stderr stream.
    """

    def __init__(self, *args, **kwargs) -> None:
        self._display_handle = None
        super().__init__(*args, **kwargs)

    def display(self, msg: str | None = None, pos: int | None = None) -> bool:
        """Render/update a single IPython display rather than print a new line."""
        from IPython.display import display

        payload = {"text/plain": str(self) if msg is None else msg}
        if self._display_handle is None:
            self._display_handle = display(payload, raw=True, display_id=True)
        else:
            self._display_handle.update(payload, raw=True)
        return True


def _select_progress() -> type:
    try:
        from IPython import get_ipython
    except ImportError:
        return TerminalProgress
    shell = get_ipython()
    if shell is None or not getattr(shell, "kernel", None):
        return TerminalProgress
    from tqdm.notebook import IProgress, tqdm as WidgetProgress
    return WidgetProgress if IProgress is not None else NotebookTextProgress


tqdm = _select_progress()
