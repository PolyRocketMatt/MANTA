from tqdm.auto import tqdm
from typing import Callable, Optional, Tuple


ProgressFn = Optional[Callable[[str], None]]
PostfixFn = Optional[Callable[..., None]]


def _get_progress(
    steps: int,
    desc: str,
) -> Tuple[ProgressFn, PostfixFn]:
    bar = tqdm(
        total=steps,
        desc=desc,
        dynamic_ncols=True
    )

    def progress(msg: str) -> None:
        bar.set_description(msg)
        bar.update(1)

    def set_postfix(**kwargs) -> None:
        bar.set_postfix(**kwargs)

    return progress, set_postfix


def _update_progress(
    progress: ProgressFn,
    message: str
) -> None:
    if progress is not None:
        progress(message)


def _update_postfix(
    postfix: PostfixFn,
    **kwargs
) -> None:
    if postfix is not None:
        postfix(**kwargs)