from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pandas.tseries.offsets import BDay
from statsmodels.tsa.statespace.structural import UnobservedComponents
from statsmodels.tools.sm_exceptions import ValueWarning
import mplcursors


warnings.filterwarnings(
    "ignore",
    message="A date index has been provided, but it has no associated frequency information.*",
    category=ValueWarning,
)
warnings.filterwarnings(
    "ignore",
    message="No supported index is available.*",
    category=ValueWarning,
)
warnings.filterwarnings(
    "ignore",
    message="No supported index is available.*",
    category=FutureWarning,
)

MODE = "short"  # "short" | "medium" | "long"

PRESETS = {
    "short":  {"HORIZON_DAYS": 15, "TRAIN_WINDOW": 90,  "ZOOM_DAYS": 180},
    "medium": {"HORIZON_DAYS": 35, "TRAIN_WINDOW": 252, "ZOOM_DAYS": 420},
    "long":   {"HORIZON_DAYS": 90, "TRAIN_WINDOW": 756, "ZOOM_DAYS": 1200},
}
if MODE not in PRESETS:
    raise ValueError(f"MODE должен быть one of {list(PRESETS)}")

HORIZON_DAYS = PRESETS[MODE]["HORIZON_DAYS"]
TRAIN_WINDOW = PRESETS[MODE]["TRAIN_WINDOW"]
ZOOM_DAYS = PRESETS[MODE]["ZOOM_DAYS"]

VOL_WINDOW = 60
LEVEL = 0.68


@dataclass
class SeriesData:
    name: str
    ticker: str
    dates: pd.DatetimeIndex
    price: pd.Series


def _latest_xlsx(raw_dir: Path) -> Path:
    files = sorted(raw_dir.glob("*.xlsx"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"Нет .xlsx в {raw_dir.resolve()}. Положи Excel туда.")
    return files[0]


def _guess_ticker_from_filename(p: Path) -> str:
    stem = p.stem
    first = stem.split("-")[0].strip()
    return first.upper() if first else stem.upper()


def _guess_name_from_filename(p: Path) -> str:
    stem = p.stem
    first = stem.split("-")[0].strip()
    return first.upper() if first else stem.upper()


def _find_col(cols: list[str], candidates: list[str]) -> str | None:
    cols_l = [c.lower() for c in cols]
    for cand in candidates:
        cand_l = cand.lower()
        for i, c in enumerate(cols_l):
            if cand_l == c:
                return cols[i]
    for cand in candidates:
        cand_l = cand.lower()
        for i, c in enumerate(cols_l):
            if cand_l in c:
                return cols[i]
    return None


def _read_excel_any(path: Path) -> tuple[SeriesData, dict]:
    df = pd.read_excel(path)
    df.columns = [str(c).strip() for c in df.columns]
    cols = df.columns.tolist()

    date_col = _find_col(cols, ["Дата", "Дата/время", "DATE", "DATETIME"])
    if not date_col:
        raise ValueError(f"Не нашел колонку даты. Колонки: {cols}")

    price_col = _find_col(
        cols,
        ["Цена, RUB", "Цена, руб", "Цена (руб)", "Цена", "Close", "Цена, USD", "USD"]
    )
    if not price_col:
        raise ValueError(f"Не нашел колонку цены. Колонки: {cols}")

    rows_raw = int(len(df))

    df2 = df[[date_col, price_col]].copy()
    df2[date_col] = pd.to_datetime(df2[date_col], errors="coerce", dayfirst=True)
    bad_dates = int(df2[date_col].isna().sum())

    def _to_num(x):
        if pd.isna(x):
            return np.nan
        s = str(x).replace("\u00a0", " ").replace(" ", "").replace(",", ".")
        s = s.replace("<", "")
        try:
            return float(s)
        except Exception:
            return np.nan

    df2[price_col] = df2[price_col].map(_to_num)
    bad_prices = int(df2[price_col].isna().sum())

    df2 = df2.dropna(subset=[date_col, price_col])
    df2 = df2.sort_values(date_col)
    dups = int(df2.duplicated(subset=[date_col]).sum())
    df2 = df2.drop_duplicates(subset=[date_col], keep="last")

    dates = pd.DatetimeIndex(df2[date_col].values)
    price = pd.Series(df2[price_col].values, index=dates, name="price")

    data = SeriesData(
        name=_guess_name_from_filename(path),
        ticker=_guess_ticker_from_filename(path),
        dates=dates,
        price=price,
    )

    diag = {
        "file": path.name,
        "date_col": date_col,
        "price_col": price_col,
        "rows_raw": rows_raw,
        "rows_after_clean": int(len(df2)),
        "bad_dates": bad_dates,
        "bad_prices": bad_prices,
        "duplicate_dates": dups,
        "date_min": str(dates.min().date()) if len(dates) else "-",
        "date_max": str(dates.max().date()) if len(dates) else "-",
    }
    return data, diag


def _use_business_days(dates: pd.DatetimeIndex) -> bool:
    if len(dates) < 20:
        return True
    diffs = np.diff(dates.values).astype("timedelta64[D]").astype(int)
    share_gt1 = (diffs > 1).mean() if len(diffs) else 0.0
    return share_gt1 > 0.15


def _volatility_daily(price: pd.Series, window: int) -> float:
    r = np.log(price).diff().dropna()
    if len(r) < 5:
        return float("nan")
    w = min(window, len(r))
    return float(r.iloc[-w:].std(ddof=0) * 100.0)


def _fit_and_forecast(price: pd.Series, horizon: int, level: float):
    model = UnobservedComponents(price, level="local linear trend")
    res = model.fit(disp=False)

    fc = res.get_forecast(steps=horizon)
    mean = fc.predicted_mean

    alpha = 1.0 - level
    ci = fc.conf_int(alpha=alpha)
    low = ci.iloc[:, 0]
    high = ci.iloc[:, 1]
    return mean, low, high


def _print_summary(diag: dict, data: SeriesData, horizon_label: str, end_low_pct: float, end_high_pct: float, vol: float):
    print("\n=== RUN SUMMARY ===")
    print(f"Файл: {diag['file']}")
    print(f"Колонки: дата='{diag['date_col']}', цена='{diag['price_col']}'")
    print(f"Строк: {diag['rows_raw']} → {diag['rows_after_clean']} (после очистки)")
    print(f"Удалено: плохие даты={diag['bad_dates']}, плохие цены={diag['bad_prices']}, дубликаты дат={diag['duplicate_dates']}")
    print(f"Даты: {diag['date_min']} .. {diag['date_max']}")
    print(f"Актив: {data.name} | тикер: {data.ticker}")
    print(f"Режим: {MODE} | окно обучения: {TRAIN_WINDOW} | горизонт: {horizon_label}")
    print(f"Волатильность({VOL_WINDOW}д): ~{vol:.2f}%/день")
    print(f"Коридор 68% на конец горизонта: {end_low_pct:+.2f}% .. {end_high_pct:+.2f}%")
    print("===================\n")


def main():
    project_root = Path(__file__).resolve().parents[1]
    raw_dir = project_root / "data" / "raw"
    reports_dir = project_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    xlsx_path = _latest_xlsx(raw_dir)
    data, diag = _read_excel_any(xlsx_path)

    y_full = data.price.copy()
    if len(y_full) < 30:
        raise ValueError("Слишком мало данных для модели (нужно хотя бы ~30 точек).")

    use_bdays = _use_business_days(y_full.index)

    train = y_full.iloc[-min(TRAIN_WINDOW, len(y_full)):]
    mean_fc, low_fc, high_fc = _fit_and_forecast(train, HORIZON_DAYS, LEVEL)

    last_date = train.index[-1]
    last_price = float(train.iloc[-1])

    if use_bdays:
        future_idx = pd.bdate_range(last_date + BDay(1), periods=HORIZON_DAYS)
        horizon_label = f"{HORIZON_DAYS} торг. дн."
    else:
        future_idx = pd.date_range(last_date + pd.Timedelta(days=1), periods=HORIZON_DAYS, freq="D")
        horizon_label = f"{HORIZON_DAYS} дн."

    mean_fc.index = future_idx
    low_fc.index = future_idx
    high_fc.index = future_idx

    forecast_line = pd.Series(
        data=np.r_[last_price, mean_fc.values],
        index=pd.DatetimeIndex([last_date]).append(mean_fc.index),
        name="forecast"
    )
    low_line = pd.Series(np.r_[last_price, low_fc.values], index=forecast_line.index)
    high_line = pd.Series(np.r_[last_price, high_fc.values], index=forecast_line.index)

    hist = y_full.iloc[-min(ZOOM_DAYS, len(y_full)):]

    vol = _volatility_daily(y_full, VOL_WINDOW)

    end_low = float(low_fc.iloc[-1])
    end_high = float(high_fc.iloc[-1])
    end_low_pct = (end_low / last_price - 1.0) * 100.0
    end_high_pct = (end_high / last_price - 1.0) * 100.0

    _print_summary(diag, data, horizon_label, end_low_pct, end_high_pct, vol)

    title = (
        f"{data.name} | MODE={MODE} | Local Linear Trend (Калман) | "
        f"обучение={TRAIN_WINDOW} | горизонт={horizon_label} | "
        f"волатильность({VOL_WINDOW}д)≈{vol:.2f}%/день"
    )

    fig = plt.figure(figsize=(18, 7), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[5.2, 1.8])

    ax = fig.add_subplot(gs[0, 0])
    ax_info = fig.add_subplot(gs[0, 1])
    ax_info.axis("off")

    l_hist, = ax.plot(hist.index, hist.values, label="Цена", linewidth=2)
    l_fc, = ax.plot(forecast_line.index, forecast_line.values, label="Прогноз (средняя траектория)", linewidth=3)
    band = ax.fill_between(
        forecast_line.index,
        low_line.values,
        high_line.values,
        alpha=0.18,
        label="Вероятностный коридор (68%)",
        linewidth=0
    )

    ax.axvline(last_date, linestyle=":", linewidth=1.5)

    ax.set_title(title)
    ax.set_xlabel("Дата")
    ax.set_ylabel("Цена")

    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=6, maxticks=10))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    ax.grid(True, alpha=0.25)
    ax.legend(handles=[l_hist, l_fc, band], loc="upper left", framealpha=0.95)

    info = (
        f"Тикер: {data.ticker}\n"
        f"Режим: {MODE}\n"
        f"Последняя цена: {last_price:.2f}\n"
        f"\n"
        f"Что означает прогноз:\n"
        f"• рыжая линия — средняя оценка\n"
        f"  направления (тренд/дрейф)\n"
        f"• синяя зона — неопределённость\n"
        f"  модели (коридор 68%)\n"
        f"\n"
        f"Коридор 68% на конец горизонта:\n"
        f"{end_low:.2f} .. {end_high:.2f}\n"
        f"({end_low_pct:+.2f}% .. {end_high_pct:+.2f}%)\n"
        f"\n"
        f"Источник: Excel (investfunds.ru)"
    )

    ax_info.text(
        0.0, 1.0, info,
        va="top", ha="left",
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="#e8f2fb", alpha=0.95),
        transform=ax_info.transAxes
    )

    cursor = mplcursors.cursor([l_hist, l_fc], hover=True)

    @cursor.connect("add")
    def _on_add(sel):
        x, yv = sel.target
        dt = mdates.num2date(x).date().isoformat()
        sel.annotation.set_text(f"{dt}\n{yv:.2f}")
        sel.annotation.get_bbox_patch().set(fc="yellow", alpha=0.9)

    out_path = reports_dir / "forecast.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")

    try:
        plt.show()
    except KeyboardInterrupt:
        print("\nОкно графика закрыто/прервано пользователем.")


if __name__ == "__main__":
    main()
