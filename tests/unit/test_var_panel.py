import pandas as pd
import pytest

from crypto_volatility.data.var_panel import (
    build_var_volatility_panel,
)


def make_asset_data(
    dates,
    values,
):
    return pd.DataFrame(
        {
            "open_time": pd.to_datetime(
                dates,
                utc=True,
            ),
            "log_gk_volatility": values,
        }
    )


def test_build_var_volatility_panel_aligns_assets():
    datasets = {
        "BTC": make_asset_data(
            [
                "2024-01-01",
                "2024-01-02",
                "2024-01-03",
            ],
            [
                1.0,
                2.0,
                3.0,
            ],
        ),
        "ETH": make_asset_data(
            [
                "2024-01-02",
                "2024-01-03",
                "2024-01-04",
            ],
            [
                10.0,
                20.0,
                30.0,
            ],
        ),
    }

    panel = build_var_volatility_panel(
        datasets=datasets,
        time_column="open_time",
    )

    assert len(panel) == 2

    assert list(
        panel.columns
    ) == [
        "open_time",
        "BTC",
        "ETH",
    ]


def test_var_panel_keeps_common_dates_only():
    datasets = {
        "BTC": make_asset_data(
            [
                "2024-01-01",
                "2024-01-02",
                "2024-01-03",
            ],
            [
                1.0,
                2.0,
                3.0,
            ],
        ),
        "ETH": make_asset_data(
            [
                "2024-01-02",
                "2024-01-03",
            ],
            [
                10.0,
                20.0,
            ],
        ),
    }

    panel = build_var_volatility_panel(
        datasets=datasets,
        time_column="open_time",
    )

    assert panel[
        "open_time"
    ].tolist() == pd.to_datetime(
        [
            "2024-01-02",
            "2024-01-03",
        ],
        utc=True,
    ).tolist()


def test_var_panel_rejects_missing_columns():
    datasets = {
        "BTC": pd.DataFrame(
            {
                "open_time": pd.to_datetime(
                    [
                        "2024-01-01",
                    ],
                    utc=True,
                )
            }
        )
    }

    with pytest.raises(
        ValueError,
        match="Missing columns",
    ):
        build_var_volatility_panel(
            datasets=datasets,
            time_column="open_time",
        )


def test_var_panel_rejects_duplicate_timestamps():
    datasets = {
        "BTC": make_asset_data(
            [
                "2024-01-01",
                "2024-01-01",
            ],
            [
                1.0,
                2.0,
            ],
        )
    }

    with pytest.raises(
        ValueError,
        match="duplicate timestamps",
    ):
        build_var_volatility_panel(
            datasets=datasets,
            time_column="open_time",
        )


def test_var_panel_rejects_empty_input():
    with pytest.raises(
        ValueError,
        match="datasets cannot be empty",
    ):
        build_var_volatility_panel(
            datasets={},
            time_column="open_time",
        )