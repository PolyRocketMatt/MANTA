import anndata as ad

from concurrent.futures import ThreadPoolExecutor
from typing import List, Literal

from ..core._samplers import (
    _sample_importance,
    _sample_stratified,
    _sample_approximate_fps
)


def sample(
    adatas: List[ad.AnnData],
    sampler: Literal["importance", "stratified", "fps"],

    fraction: float = 0.1,
    bin_size: int = 256,
    spatial_key: str = "spatial_manta",
    sampling_key: str = "sampling",
    gamma: float = 1.0,
    shuffle: bool = True,
) -> None:
    def _sample(adata: ad.AnnData):
        match sampler:
            case "importance":
                _sample_importance(
                    adata=adata,
                    fraction=fraction,
                    bin_size=bin_size,
                    spatial_key=spatial_key,
                    sample_key=sampling_key,
                    gamma=gamma
                )
            case "stratified":
                _sample_stratified(
                    adata=adata,
                    bin_size=bin_size,
                    spatial_key=spatial_key,
                    sample_key=sampling_key,
                    shuffle=shuffle
                )
            case "fps":
                _sample_approximate_fps(
                    adata=adata,
                    fraction=fraction,
                    bin_size=bin_size,
                    spatial_key=spatial_key,
                    sample_key=sampling_key,
                    shuffle=shuffle
                )
            case _:
                raise ValueError(
                    f"`{sampler}` is not a valid sampler"
                )

    for adata in adatas:
        _sample(adata)

    """
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                _sample,
                adata=adata,
            )
            for adata in adatas
        ]

        for future in futures:
            future.result()
    """