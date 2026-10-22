from __future__ import annotations

import numpy as np
import pandas as pd


def build_var_volatility_panel(
    datasets: dict[str, pd.DataFrame],
    time_column: str,
    volatility_column: str = "log_gk_volatility",
) -> pd.DataFrame:
    """
    Align multiple asset volatility series into one VAR panel.

    The output contains one row per common timestamp and one
    volatility column per asset.
    """
    if not datasets:
        raise ValueError(
            "datasets cannot be empty"
        )

    frames = []

    for asset, df in datasets.items():
        required_columns = {
            time_column,
            volatility_column,
        }

        missing_columns = (
            required_columns
            - set(df.columns)
        )

        if missing_columns:
            raise ValueError(
                f"Missing columns for {asset}: "
                f"{sorted(missing_columns)}"
            )

        asset_frame = (
            df[
                [
                    time_column,
                    volatility_column,
                ]
            ]
            .copy()
            .rename(
                columns={
                    volatility_column: asset,
                }
            )
        )

        if asset_frame[
            time_column
        ].duplicated().any():
            raise ValueError(
                f"{asset} contains duplicate timestamps"
            )

        frames.append(
            asset_frame
        )

    panel = frames[
        0
    ]

    for frame in frames[
        1:
    ]:
        panel = panel.merge(
            frame,
            on=time_column,
            how="inner",
        )

    panel = (
        panel
        .sort_values(
            time_column
        )
        .reset_index(
            drop=True
        )
    )

    asset_columns = list(
        datasets.keys()
    )

    if panel.empty:
        raise ValueError(
            "No common timestamps found across assets"
        )

    if panel[
        asset_columns
    ].isna().any().any():
        raise ValueError(
            "VAR panel cannot contain missing volatility values"
        )

    if not np.isfinite(
        panel[
            asset_columns
        ].to_numpy(
            dtype=float
        )
    ).all():
        raise ValueError(
            "VAR panel volatility values must be finite"
        )

    return panel