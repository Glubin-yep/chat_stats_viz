#!/usr/bin/env python3
"""
chat_stats_viz.py
Аналізує export Telegram (result.json) і генерує інфографіку:
- Хто скільки разів перший писав (після перерви)
- Середній та медіанний час відповіді по кожному співрозмовнику
- Гістограма часів відповіді та барчарт стартів

Вхід: result.json (структура як у Telegram-export)
"""

import json
import os
from collections import defaultdict, Counter
from datetime import datetime, timezone
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- Налаштування ----------
INPUT_FILE = "result.json"
OUTPUT_DIR = "output"
# Перерва (в годинах), після якої наступне повідомлення вважається "новим стартом"
INACTIVITY_THRESHOLD_HOURS = 6.0

# Мінімальний та максимальний час відповіді, які будемо враховувати (в секундах)
MAX_RESPONSE_SECONDS = 60 * 60 * 24 * 7  # 7 днів (ігноруємо надто великі паузи як "не відповідь")
MIN_RESPONSE_SECONDS = 1  # мінімум 1 секунда (ігноруємо нульові або негативні інтервали якщо знайдеться)

# ---------- Допоміжні функції ----------
def load_messages(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    msgs = data.get("messages", [])
    # В деяких експортів час в полі date_unixtime (string) — приведемо до int
    parsed = []
    for m in msgs:
        if "date_unixtime" in m and m["date_unixtime"]:
            ts = int(m["date_unixtime"])
        else:
            # fallback: парсити date (ISO)
            ts = int(datetime.fromisoformat(m["date"]).replace(tzinfo=timezone.utc).timestamp())
        parsed.append({
            "id": m.get("id"),
            "from": m.get("from"),
            "from_id": m.get("from_id"),
            "ts": ts,
            "type": m.get("type"),
            "text": m.get("text", "") if m.get("text") is not None else "",
            "media": any(k in m for k in ("photo", "file", "media_type", "video", "voice_message"))
        })
    # відсортуємо за часом
    parsed.sort(key=lambda x: x["ts"])
    return parsed

def human_readable_seconds(s):
    if s is None or math.isnan(s):
        return "-"
    s = int(round(s))
    parts = []
    days, rem = divmod(s, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, seconds = divmod(rem, 60)
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    if seconds:
        parts.append(f"{seconds}s")
    return " ".join(parts) if parts else "0s"

# ---------- Аналіз ----------
def analyze(messages):
    users = sorted(
    {str(m["from"] or m["from_id"]) for m in messages if m.get("from") or m.get("from_id")}
)

    # Якщо поле "from" пусте (system messages), ігноруємо такі повідомлення у статистиці відповіді/стартів
    # Але збережемо для логу
    valid_msgs = [m for m in messages if m["from"]]
    inactivity_threshold = INACTIVITY_THRESHOLD_HOURS * 3600

    # Хто скільки разів першим писав (за визначенням — перше повідомлення or повідомлення після перерви)
    starter_counts = Counter()
    # Лог стартів із таймштампом
    starters = []

    prev_ts = None
    for i, m in enumerate(valid_msgs):
        if prev_ts is None or (m["ts"] - prev_ts) > inactivity_threshold:
            starter_counts[m["from"]] += 1
            starters.append((m["from"], m["ts"]))
        prev_ts = m["ts"]

    # Секції для time-to-reply:
    # Для кожного переходу X -> Y (X != Y) беремо delta = ts(Y) - ts(X) і додаємо його до відповідача Y
    response_times_by_user = defaultdict(list)
    # Перебираємо послідовні повідомлення і фіксуємо тільки коли автор змінюється
    for i in range(1, len(valid_msgs)):
        prev = valid_msgs[i-1]
        cur = valid_msgs[i]
        if prev["from"] and cur["from"] and prev["from"] != cur["from"]:
            delta = cur["ts"] - prev["ts"]
            if MIN_RESPONSE_SECONDS <= delta <= MAX_RESPONSE_SECONDS:
                response_times_by_user[cur["from"]].append(delta)

    # Підрахунки: середнє, медіана, кількість
    stats = []
    for u in users:
        times = response_times_by_user.get(u, [])
        if times:
            arr = np.array(times)
            mean = float(arr.mean())
            median = float(np.median(arr))
            p90 = float(np.percentile(arr, 90))
            cnt = len(arr)
        else:
            mean = median = p90 = float("nan")
            cnt = 0
        stats.append({
            "user": u,
            "responses_count": cnt,
            "mean_response_seconds": mean,
            "median_response_seconds": median,
            "p90_response_seconds": p90,
            "starter_count": starter_counts.get(u, 0),
        })

    stats_df = pd.DataFrame(stats).sort_values(by="responses_count", ascending=False)
    return {
        "starter_counts": starter_counts,
        "starters": starters,
        "response_times_by_user": response_times_by_user,
        "stats_df": stats_df,
        "total_messages": len(valid_msgs),
        "users": users
    }

# ---------- Візуалізація ----------
def visualize(results, outdir):
    os.makedirs(outdir, exist_ok=True)
    stats_df = results["stats_df"]
    starter_counts = results["starter_counts"]
    response_times_by_user = results["response_times_by_user"]
    users = results["users"]

    # 1) Bar chart — хто скільки разів стартував
    plt.figure(figsize=(8,4))
    users_for_bar = list(starter_counts.keys())
    counts = [starter_counts[u] for u in users_for_bar]
    plt.bar(users_for_bar, counts)
    plt.title("Кількість стартів розмови (перерва >= {} год)".format(INACTIVITY_THRESHOLD_HOURS))
    plt.ylabel("Кількість стартів")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "starters_bar.png"))
    plt.close()

    # 2) Bar chart — середній час відповіді (в хвилинах)
    df_plot = stats_df.copy()
    df_plot["mean_min"] = df_plot["mean_response_seconds"] / 60.0
    plt.figure(figsize=(8,4))
    plt.bar(df_plot["user"], df_plot["mean_min"])
    plt.title("Середній час відповіді (хвилини)")
    plt.ylabel("Хвилини")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "mean_response_minutes_bar.png"))
    plt.close()

    # 3) Boxplot/гістограма часів відповіді для кожного користувача
    # Зберемо дані у форматі list
    all_data = []
    labels = []
    for u in users:
        times = response_times_by_user.get(u, [])
        if times:
            all_data.append(np.array(times) / 60.0)  # хвилини
            labels.append(u)
    if all_data:
        plt.figure(figsize=(10,6))
        sns.boxplot(data=all_data)
        plt.gca().set_xticklabels(labels, rotation=45, ha="right")
        plt.title("Розподіл часів відповіді (хвилини) — boxplot")
        plt.ylabel("Хвилини")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "response_times_boxplot.png"))
        plt.close()

    # 4) Табличний CSV з підсумками
    stats_df.to_csv(os.path.join(outdir, "summary_stats.csv"), index=False)

# ---------- Головна логіка ----------
def main():
    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: input file '{INPUT_FILE}' not found.")
        return
    msgs = load_messages(INPUT_FILE)
    print(f"Loaded {len(msgs)} messages (with 'from' fields considered).")

    results = analyze(msgs)
    print("\n--- Підсумкова таблиця (top) ---")
    pd.set_option("display.max_rows", 20)
    print(results["stats_df"].to_string(index=False, 
        formatters={
            "mean_response_seconds": lambda x: human_readable_seconds(x) if not math.isnan(x) else "-",
            "median_response_seconds": lambda x: human_readable_seconds(x) if not math.isnan(x) else "-",
            "p90_response_seconds": lambda x: human_readable_seconds(x) if not math.isnan(x) else "-",
        }
    ))

    print("\nСписок стартів (пара: користувач — час):")
    for u, ts in results["starters"][:50]:
        print(f"  - {u} @ {datetime.fromtimestamp(ts).astimezone().isoformat()}")

    # Візуалізація
    visualize(results, OUTPUT_DIR)
    print(f"\nГрафіки та CSV збережено в папці '{OUTPUT_DIR}'.")

if __name__ == "__main__":
    main()
