"""Decision-time and bar-availability rules for causal market sequences."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, date, datetime

from src.prediction.development_contract import CausalDevelopmentContract
from src.prediction.timeseries.contract import TimeseriesModelContract


def _require_aware(value: datetime, *, field: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    return value


def daily_v1_prediction_as_of(
    published_date: date,
    *,
    development: CausalDevelopmentContract,
) -> datetime:
    """Return the conservative date-only pre-open decision timestamp."""
    return datetime.combine(
        published_date,
        development.decision_time,
        tzinfo=development.zone,
    )


def next_trading_preopen_prediction_as_of(
    published_date: date,
    *,
    trading_days: Iterable[date],
    development: CausalDevelopmentContract,
) -> datetime:
    """Return the first trading-session pre-open strictly after the date label.

    Historical announcement timestamps are only date-precise.  The LSTM branch
    therefore keeps market inputs strictly before ``published_date`` while using
    the following trading session's 08:30 pre-open as the shared component/fusion
    prediction timestamp.
    """
    candidates = sorted({day for day in trading_days if day > published_date})
    if not candidates:
        raise ValueError(
            "trading calendar has no session strictly after publication date"
        )
    return datetime.combine(
        candidates[0],
        development.decision_time,
        tzinfo=development.zone,
    )


def daily_bar_available_at(
    trade_date: date,
    *,
    contract: TimeseriesModelContract,
    development: CausalDevelopmentContract,
) -> datetime:
    """Timestamp a completed daily bar after the configured market close delay."""
    return datetime.combine(
        trade_date,
        contract.daily_bar_available_time,
        tzinfo=development.zone,
    )


def require_daily_v1_bar_usable(
    *,
    trade_date: date,
    published_date: date,
    prediction_as_of: datetime,
    bar_available_at: datetime,
    development: CausalDevelopmentContract,
) -> None:
    """Reject same-day/future bars and bars unavailable at the decision time."""
    prediction = _require_aware(prediction_as_of, field="prediction_as_of")
    available = _require_aware(bar_available_at, field="bar_available_at")
    if not development.historical_daily_bar_allowed(
        trade_date=trade_date,
        announcement_published_date=published_date,
    ):
        raise ValueError(
            "same-day or future daily bar rejected: "
            f"{trade_date.isoformat()} versus publication {published_date.isoformat()}"
        )
    development.require_available(
        feature_name=f"daily_bar:{trade_date.isoformat()}",
        available_at=available,
        prediction_as_of=prediction,
    )


def first_authoritative_preopen_decision(
    published_at: datetime,
    *,
    trading_days: Iterable[date],
    publication_time_authoritative: bool,
    development: CausalDevelopmentContract,
) -> datetime:
    """Resolve timing-v2 only when an authoritative publication time exists.

    The supplied trading calendar is explicit so weekends and holidays cannot be
    guessed from wall-clock dates. A publication after the day's pre-open decision,
    including an after-close disclosure, moves to the next trading-day decision.
    """
    if not publication_time_authoritative:
        raise PermissionError(
            "intraday timing is blocked without an authoritative publication timestamp"
        )
    publication = _require_aware(published_at, field="published_at").astimezone(
        development.zone
    )
    for trading_day in sorted(set(trading_days)):
        candidate = datetime.combine(
            trading_day,
            development.decision_time,
            tzinfo=development.zone,
        )
        if candidate >= publication:
            return candidate
    raise ValueError("trading calendar has no decision after publication")


def require_timestamped_bar_available(
    *,
    bar_available_at: datetime,
    prediction_as_of: datetime,
    feature_name: str = "timestamped_bar",
) -> None:
    """Generic timing-v2 guard for intraday or completed daily bars."""
    available = _require_aware(bar_available_at, field="bar_available_at").astimezone(
        UTC
    )
    decision = _require_aware(prediction_as_of, field="prediction_as_of").astimezone(
        UTC
    )
    if available > decision:
        raise ValueError(
            f"future bar rejected: {feature_name} available at "
            f"{available.isoformat()} after as_of {decision.isoformat()}"
        )
