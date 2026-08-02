from tqdm.auto import tqdm
from typing import Callable, Optional


ProgressFn = Optional[Callable[[str], None]]


def _get_progress(
    steps: int,
    desc: str,
) -> ProgressFn:
    steps = tqdm(
        total=5,
        desc="Preprocessing",
        dynamic_ncols=True
    )

    def progress(msg: str) -> None:
        steps.set_description(msg)
        steps.update(1)

    return progress


def _update_progress(
    progress: ProgressFn,
    message: str
) -> None:
    if progress is not None:
        progress(message)