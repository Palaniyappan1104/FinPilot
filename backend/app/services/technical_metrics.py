"""Deterministic technical metrics and indicator calculation engine for FinPilot.

Phase 7.2 implements pure Python technical analysis algorithms:
- SMA (Simple Moving Average) across 20, 50, 200 periods.
- EMA (Exponential Moving Average) across 20, 50, 200, 12, 26 periods with SMA seeding.
- RSI (Wilder's 14-period Relative Strength Index) with zero-loss/gain edge handling.
- MACD (12/26/9 Fast line, Signal line, and Histogram).
- Volume analysis (latest volume, 20-period rolling average, volume ratio).
- Support and resistance level detection via rolling swing extrema.
- Categorical trend detection ('uptrend', 'downtrend', 'sideways').
- Deterministic 0-100 technical scoring function with component breakdown.

Boundary Rules:
- Consumes HistoricalMarketData from Phase 7.1 as read-only input.
- Purely deterministic; no network, API, or LLM dependencies.
- Preserves full float precision without premature rounding.
- Insufficient history produces None rather than fabricated or guessed values.
"""

import math
from datetime import datetime, timezone
from typing import List, Optional, Tuple

from app.core.logging import get_logger
from app.models.market_data import HistoricalMarketData, OHLCVCandle
from app.models.technical_metrics import (
    MACDMetrics,
    MovingAverageMetrics,
    RSIMetrics,
    SupportResistanceMetrics,
    TechnicalMetrics,
    TechnicalScoreBreakdown,
    TrendDirection,
    VolumeMetrics,
)

logger = get_logger("app.services.technical_metrics")


def safe_divide(
    numerator: Optional[float], denominator: Optional[float]
) -> Optional[float]:
    """Safely divide two numbers, returning None on zero division or invalid input."""
    if numerator is None or denominator is None:
        return None
    try:
        num = float(numerator)
        denom = float(denominator)
        if (
            denom == 0.0
            or math.isnan(num)
            or math.isnan(denom)
            or math.isinf(num)
            or math.isinf(denom)
        ):
            return None
        res = num / denom
        if math.isnan(res) or math.isinf(res):
            return None
        return res
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def calculate_sma_series(prices: List[float], period: int) -> List[Optional[float]]:
    """Compute rolling Simple Moving Average series.

    Args:
        prices: Chronological sequence of prices.
        period: Rolling window length (e.g. 20, 50, 200).

    Returns:
        List of same length as prices, containing float SMA values where
        sufficient history exists and None elsewhere.
    """
    if period <= 0:
        raise ValueError(f"SMA period must be strictly positive, got {period}.")

    result: List[Optional[float]] = [None] * len(prices)
    if len(prices) < period:
        return result

    window_sum = sum(prices[:period])
    result[period - 1] = window_sum / float(period)

    for i in range(period, len(prices)):
        window_sum += prices[i] - prices[i - period]
        result[i] = window_sum / float(period)

    return result


def calculate_sma(prices: List[float], period: int) -> Optional[float]:
    """Compute latest Simple Moving Average value."""
    if len(prices) < period or period <= 0:
        return None
    return sum(prices[-period:]) / float(period)


def calculate_ema_series(prices: List[float], period: int) -> List[Optional[float]]:
    """Compute Exponential Moving Average series using standard SMA seeding.

    Methodology:
    - Multiplier alpha = 2.0 / (period + 1.0).
    - Seed: The first 'period' observations are averaged via SMA to form EMA_{period-1}.
    - Recurrence: EMA_i = Price_i * alpha + EMA_{i-1} * (1.0 - alpha).

    Args:
        prices: Chronological sequence of prices.
        period: EMA window length (e.g. 12, 20, 26, 50, 200).

    Returns:
        List of same length as prices, with EMA values and None where history < period.
    """
    if period <= 0:
        raise ValueError(f"EMA period must be strictly positive, got {period}.")

    result: List[Optional[float]] = [None] * len(prices)
    if len(prices) < period:
        return result

    alpha = 2.0 / (float(period) + 1.0)
    sma_seed = sum(prices[:period]) / float(period)
    result[period - 1] = sma_seed

    current_ema = sma_seed
    one_minus_alpha = 1.0 - alpha
    for i in range(period, len(prices)):
        current_ema = prices[i] * alpha + current_ema * one_minus_alpha
        result[i] = current_ema

    return result


def calculate_ema(prices: List[float], period: int) -> Optional[float]:
    """Compute latest Exponential Moving Average value."""
    series = calculate_ema_series(prices, period)
    return series[-1] if series else None


def calculate_rsi_series(
    prices: List[float], period: int = 14
) -> List[Optional[float]]:
    """Compute Relative Strength Index series using Wilder's smoothing.

    Methodology:
    - Price changes Delta_i = Price_i - Price_{i-1}.
    - Gains U_i = max(Delta_i, 0), Losses D_i = max(-Delta_i, 0).
    - Initial averages at index 'period': Simple mean of gains and losses.
    - Subsequent smoothing: Avg = (Prior_Avg * (period - 1) + Current) / period.
    - Flat series (0 loss, 0 gain) resolves to 50.0.
    - 0 loss with positive gain resolves to 100.0.
    - 0 gain with positive loss resolves to 0.0.

    Args:
        prices: Chronological sequence of closing prices.
        period: Lookback window (standard default: 14).

    Returns:
        List of length equal to prices, with RSI values in [0.0, 100.0] or None.
    """
    if period <= 0:
        raise ValueError(f"RSI period must be strictly positive, got {period}.")

    n = len(prices)
    result: List[Optional[float]] = [None] * n
    # Requires at least period + 1 prices to form period price changes
    if n <= period:
        return result

    gains: List[float] = [0.0] * n
    losses: List[float] = [0.0] * n
    for i in range(1, n):
        diff = prices[i] - prices[i - 1]
        if diff > 0:
            gains[i] = diff
        elif diff < 0:
            losses[i] = -diff

    avg_gain = sum(gains[1 : period + 1]) / float(period)
    avg_loss = sum(losses[1 : period + 1]) / float(period)

    def _compute_rsi(g: float, l_val: float) -> float:
        if l_val == 0.0:
            return 50.0 if g == 0.0 else 100.0
        if g == 0.0:
            return 0.0
        rs = g / l_val
        return 100.0 - (100.0 / (1.0 + rs))

    result[period] = _compute_rsi(avg_gain, avg_loss)

    factor = float(period - 1)
    for i in range(period + 1, n):
        avg_gain = (avg_gain * factor + gains[i]) / float(period)
        avg_loss = (avg_loss * factor + losses[i]) / float(period)
        result[i] = _compute_rsi(avg_gain, avg_loss)

    return result


def calculate_rsi(prices: List[float], period: int = 14) -> Optional[float]:
    """Compute latest Wilder RSI value."""
    series = calculate_rsi_series(prices, period)
    return series[-1] if series else None


def calculate_macd(
    prices: List[float],
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> MACDMetrics:
    """Compute Moving Average Convergence Divergence indicators.

    Methodology:
    - MACD Line = EMA(fast_period) - EMA(slow_period).
    - Signal Line = EMA(signal_period) of MACD Line (seeded with SMA of MACD).
    - Histogram = MACD Line - Signal Line.

    Requires at least (slow_period + signal_period - 1) observations.
    """
    metrics = MACDMetrics(
        fast_period=fast_period,
        slow_period=slow_period,
        signal_period=signal_period,
    )

    if len(prices) < slow_period:
        return metrics

    ema_fast = calculate_ema_series(prices, fast_period)
    ema_slow = calculate_ema_series(prices, slow_period)

    macd_series: List[Optional[float]] = [None] * len(prices)
    for i in range(slow_period - 1, len(prices)):
        f_val = ema_fast[i]
        s_val = ema_slow[i]
        if f_val is not None and s_val is not None:
            macd_series[i] = f_val - s_val

    latest_macd = macd_series[-1]
    metrics.macd_line = latest_macd

    # Extract valid contiguous MACD values for signal line computation
    valid_macd = [v for v in macd_series if v is not None]
    if len(valid_macd) < signal_period or latest_macd is None:
        return metrics

    signal_series = calculate_ema_series(valid_macd, signal_period)
    latest_signal = signal_series[-1]
    metrics.signal_line = latest_signal

    if latest_signal is not None:
        metrics.histogram = latest_macd - latest_signal

    return metrics


def calculate_volume_metrics(volumes: List[float], window: int = 20) -> VolumeMetrics:
    """Compute latest volume, rolling average volume, and volume ratio."""
    if not volumes:
        return VolumeMetrics()

    latest_vol = volumes[-1]
    if len(volumes) < window or window <= 0:
        return VolumeMetrics(latest_volume=latest_vol)

    avg_vol = sum(volumes[-window:]) / float(window)
    vol_ratio: Optional[float] = None
    if avg_vol > 0.0 and not math.isnan(avg_vol) and not math.isinf(avg_vol):
        vol_ratio = latest_vol / avg_vol

    return VolumeMetrics(
        latest_volume=latest_vol,
        average_volume_20d=avg_vol,
        volume_ratio=vol_ratio,
    )


def calculate_support_resistance(
    candles: List[OHLCVCandle],
    lookback: int = 50,
    current_price: Optional[float] = None,
) -> SupportResistanceMetrics:
    """Detect deterministic price support and resistance levels from extrema.

    Methodology:
    - Evaluates the most recent 'lookback' candles (minimum 5 candles required).
    - Identifies swing lows (L_i <= L_{i-1}, L_i <= L_{i+1}) and swing highs.
    - Filters support levels as lows strictly below current close (descending).
    - Filters resistance levels as highs strictly above current close (ascending).
    - Falls back to lookback extremes if no strict swing points exist.
    """
    if len(candles) < 5:
        return SupportResistanceMetrics()

    window_candles = candles[-lookback:]
    latest_close = current_price if current_price is not None else candles[-1].close

    swing_lows: List[float] = []
    swing_highs: List[float] = []

    # Swing detection with 1-bar left/right neighborhood
    for i in range(1, len(window_candles) - 1):
        c_prev = window_candles[i - 1]
        c_curr = window_candles[i]
        c_next = window_candles[i + 1]

        if c_curr.low <= c_prev.low and c_curr.low <= c_next.low:
            swing_lows.append(c_curr.low)
        if c_curr.high >= c_prev.high and c_curr.high >= c_next.high:
            swing_highs.append(c_curr.high)

    # Window extremes as bounds
    window_min = min(c.low for c in window_candles)
    window_max = max(c.high for c in window_candles)

    # Dedup helper within 0.05% threshold to avoid clustering
    def _dedup_levels(levels: List[float]) -> List[float]:
        unique: List[float] = []
        for val in sorted(levels):
            if not any(abs(val - u) / u < 0.0005 for u in unique if u > 0):
                unique.append(val)
        return unique

    candidate_supports = [l_val for l_val in swing_lows if l_val < latest_close]
    if not candidate_supports and window_min < latest_close:
        candidate_supports.append(window_min)

    candidate_resistances = [h_val for h_val in swing_highs if h_val > latest_close]
    if not candidate_resistances and window_max > latest_close:
        candidate_resistances.append(window_max)

    # Deduplicate and sort
    deduped_sup = _dedup_levels(candidate_supports)
    deduped_res = _dedup_levels(candidate_resistances)

    # Supports: sorted descending so index 0 is nearest to current price
    sorted_supports = sorted(deduped_sup, reverse=True)
    # Resistances: sorted ascending so index 0 is nearest to current price
    sorted_resistances = sorted(deduped_res)

    primary_sup = sorted_supports[0] if sorted_supports else None
    primary_res = sorted_resistances[0] if sorted_resistances else None

    # Handle edge case where market is completely flat
    if primary_sup is None and window_min <= latest_close:
        primary_sup = window_min
    if primary_res is None and window_max >= latest_close:
        primary_res = window_max

    return SupportResistanceMetrics(
        primary_support=primary_sup,
        primary_resistance=primary_res,
        support_levels=sorted_supports,
        resistance_levels=sorted_resistances,
    )


def detect_trend(
    close: Optional[float],
    sma_20: Optional[float],
    sma_50: Optional[float],
    sma_200: Optional[float] = None,
) -> Optional[TrendDirection]:
    """Classify price trend using moving average alignment and price action.

    Rules:
    - If SMA20 and SMA50 are available:
        - If SMA200 is available:
            - SMA20 > SMA50 > SMA200 and close >= SMA20 -> 'uptrend'
            - SMA20 < SMA50 < SMA200 and close <= SMA20 -> 'downtrend'
            - Otherwise -> 'sideways'
        - If SMA200 is unavailable (20 <= T < 200):
            - SMA20 > SMA50 and close >= SMA20 -> 'uptrend'
            - SMA20 < SMA50 and close <= SMA20 -> 'downtrend'
            - Otherwise -> 'sideways'
    - Insufficient history (T < 20 or missing inputs) -> None.
    """
    if close is None or sma_20 is None or sma_50 is None:
        return None

    if sma_200 is not None:
        if sma_20 > sma_50 > sma_200 and close >= sma_20:
            return "uptrend"
        if sma_20 < sma_50 < sma_200 and close <= sma_20:
            return "downtrend"
        return "sideways"

    # Fallback to 2-MA alignment when SMA200 is not yet formed
    if sma_20 > sma_50 and close >= sma_20:
        return "uptrend"
    if sma_20 < sma_50 and close <= sma_20:
        return "downtrend"
    return "sideways"


def calculate_technical_score(
    close: Optional[float],
    mas: MovingAverageMetrics,
    rsi: RSIMetrics,
    macd: MACDMetrics,
    vol: VolumeMetrics,
) -> Tuple[Optional[float], Optional[TechnicalScoreBreakdown]]:
    """Compute a transparent, bounded deterministic technical score in [0.0, 100.0].

    Components (25.0 points each):
    1. Moving Average Alignment: Stacking order relative to SMA20/50/200.
    2. RSI Momentum Regime: Bullish zone (50-70) scores highest; oversold lowest.
    3. MACD Relationship: MACD vs Signal and histogram sign.
    4. Volume Confirmation: Expanding volume confirming price momentum.

    Data-Sufficiency Safety:
    Requires at least 3 pillars (available_points >= 75.0) to produce a composite
    score. If fewer pillars are computable, returns (None, None) to prevent
    misleading 0-100 scores derived from sparse partial data.
    """
    if close is None:
        return None, None

    breakdown = TechnicalScoreBreakdown()
    available_points = 0.0
    earned_points = 0.0

    # 1. Moving Average Alignment (25 pts)
    if mas.sma_20 is not None and mas.sma_50 is not None:
        available_points += 25.0
        if mas.sma_200 is not None:
            if close >= mas.sma_20 > mas.sma_50 > mas.sma_200:
                s_ma = 25.0
            elif mas.sma_20 > mas.sma_50 > mas.sma_200:
                s_ma = 20.0
            elif close >= mas.sma_20 and mas.sma_20 > mas.sma_50:
                s_ma = 17.5
            elif close >= mas.sma_20 or mas.sma_20 > mas.sma_50:
                s_ma = 12.5
            elif close <= mas.sma_20 < mas.sma_50 < mas.sma_200:
                s_ma = 0.0
            elif mas.sma_20 < mas.sma_50 < mas.sma_200:
                s_ma = 5.0
            else:
                s_ma = 10.0
        else:
            if close >= mas.sma_20 > mas.sma_50:
                s_ma = 25.0
            elif close >= mas.sma_20:
                s_ma = 17.5
            elif close <= mas.sma_20 < mas.sma_50:
                s_ma = 0.0
            elif close <= mas.sma_20:
                s_ma = 7.5
            else:
                s_ma = 12.5
        breakdown.ma_alignment_score = s_ma
        earned_points += s_ma

    # 2. RSI Momentum Regime (25 pts)
    if rsi.rsi_14 is not None:
        available_points += 25.0
        r = rsi.rsi_14
        if 50.0 <= r <= 70.0:
            s_rsi = 25.0
        elif r > 70.0:
            s_rsi = 15.0  # Bullish momentum with overbought caution
        elif 40.0 <= r < 50.0:
            s_rsi = 12.5  # Neutral consolidation
        elif 30.0 <= r < 40.0:
            s_rsi = 5.0  # Weak / bearish
        else:
            s_rsi = 0.0  # Deeply oversold / distressed
        breakdown.rsi_score = s_rsi
        earned_points += s_rsi

    # 3. MACD Relationship (25 pts)
    if macd.macd_line is not None and macd.signal_line is not None:
        available_points += 25.0
        hist_pos = macd.histogram is not None and macd.histogram > 0.0
        hist_neg = macd.histogram is not None and macd.histogram < 0.0

        if macd.macd_line > macd.signal_line and hist_pos:
            s_macd = 25.0
        elif macd.macd_line > macd.signal_line:
            s_macd = 17.5
        elif macd.macd_line <= macd.signal_line and hist_neg:
            s_macd = 0.0
        elif macd.macd_line <= macd.signal_line:
            s_macd = 7.5
        else:
            s_macd = 12.5
        breakdown.macd_score = s_macd
        earned_points += s_macd

    # 4. Volume Confirmation (25 pts)
    if vol.volume_ratio is not None:
        available_points += 25.0
        vr = vol.volume_ratio
        sma_ref = mas.sma_20 or close
        price_above = close >= sma_ref

        if vr >= 1.2:
            s_vol = 25.0 if price_above else 0.0
        elif vr >= 1.0:
            s_vol = 20.0 if price_above else 5.0
        elif vr >= 0.8:
            s_vol = 12.5
        else:
            s_vol = 10.0
        breakdown.volume_score = s_vol
        earned_points += s_vol

    # Require at least 3 pillars (75.0 points) for data sufficiency
    if available_points < 75.0:
        return None, None

    # Normalize to 100-point scale based on available indicator pillars
    normalized_score = (earned_points / available_points) * 100.0
    bounded_score = max(0.0, min(100.0, normalized_score))
    return bounded_score, breakdown


def calculate_technical_metrics(
    market_data: HistoricalMarketData,
) -> TechnicalMetrics:
    """Compute standardized technical indicators from historical market data.

    Pure deterministic Python execution without LLM or external network dependencies.
    Does not mutate the incoming HistoricalMarketData instance.

    Args:
        market_data: Normalized HistoricalMarketData from Phase 7.1.

    Returns:
        TechnicalMetrics: Strongly typed container with moving averages, RSI,
        MACD, volume analytics, support/resistance levels, trend, and score.
    """
    ticker = market_data.ticker
    candles = market_data.candles or []

    if not candles:
        raise ValueError(
            f"Cannot calculate technical metrics for ticker '{ticker}': "
            "candle list is empty."
        )

    latest_close = candles[-1].close
    close_prices = [c.close for c in candles]
    volumes = [c.volume for c in candles]

    # 1. Moving Averages (SMA & EMA)
    mas = MovingAverageMetrics(
        sma_20=calculate_sma(close_prices, 20),
        sma_50=calculate_sma(close_prices, 50),
        sma_200=calculate_sma(close_prices, 200),
        ema_20=calculate_ema(close_prices, 20),
        ema_50=calculate_ema(close_prices, 50),
        ema_200=calculate_ema(close_prices, 200),
        ema_12=calculate_ema(close_prices, 12),
        ema_26=calculate_ema(close_prices, 26),
    )

    # 2. RSI (Wilder 14)
    rsi = RSIMetrics(
        rsi_14=calculate_rsi(close_prices, 14),
        period=14,
    )

    # 3. MACD (12, 26, 9)
    macd = calculate_macd(close_prices, fast_period=12, slow_period=26, signal_period=9)

    # 4. Volume Analysis
    vol = calculate_volume_metrics(volumes, window=20)

    # 5. Support and Resistance Levels
    sr = calculate_support_resistance(candles, lookback=50)

    # 6. Trend Detection
    trend = detect_trend(latest_close, mas.sma_20, mas.sma_50, mas.sma_200)

    # 7. Technical Score
    tech_score, score_breakdown = calculate_technical_score(
        latest_close, mas, rsi, macd, vol
    )

    return TechnicalMetrics(
        ticker=ticker,
        latest_close=latest_close,
        calculated_at=datetime.now(timezone.utc),
        candle_count=len(candles),
        moving_averages=mas,
        rsi=rsi,
        macd=macd,
        volume=vol,
        support_resistance=sr,
        trend=trend,
        technical_score=tech_score,
        score_breakdown=score_breakdown,
    )
