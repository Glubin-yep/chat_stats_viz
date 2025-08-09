#!/usr/bin/env python3
"""
Розширений аналіз чату з Telegram та Instagram
"""

import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

def safe_filename(name):
    """Створює безпечне ім'я файлу, видаляючи неприпустимі символи."""
    return re.sub(r'[^a-zA-Z0-9а-яА-ЯёЁіІїЇєЄ _-]', '_', name)

# ---------- Налаштування ----------
INPUT_FILE = "message_1.json"  # Може бути "result.json" (Telegram) або "message_1.json" (Instagram)
OUTPUT_DIR = "output"
INACTIVITY_THRESHOLD_HOURS = 6.0
MAX_RESPONSE_SECONDS = 60 * 60 * 24 * 7
MIN_RESPONSE_SECONDS = 1

STOPWORDS = set("""
і в на що з до це за як але чи а та ж або ну ось вже від бо
""".split())

# ---------- Завантаження ----------
def load_messages(paths):
    """Завантажує та об'єднує повідомлення з кількох файлів JSON."""
    parsed_all = []

    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Формат Instagram
        if "participants" in data and "messages" in data:
            for m in data.get("messages", []):
                parsed_all.append({
                    "id": None,
                    "from": m.get("sender_name"),
                    "from_id": m.get("sender_name"),
                    "ts": int(m["timestamp_ms"] / 1000),
                    "text": m.get("content", ""),
                    "is_media": any(k in m for k in ("photos", "videos", "audio_files", "share"))
                })
        # Формат Telegram
        else:
            msgs = data.get("messages", [])
            for m in msgs:
                if "date_unixtime" in m and m["date_unixtime"]:
                    ts = int(m["date_unixtime"])
                else:
                    ts = int(datetime.fromisoformat(m["date"]).replace(tzinfo=timezone.utc).timestamp())
                parsed_all.append({
                    "id": m.get("id"),
                    "from": m.get("from"),
                    "from_id": m.get("from_id"),
                    "ts": ts,
                    "text": m.get("text", ""),
                    "is_media": any(k in m for k in ("photo", "file", "media_type", "video", "voice_message"))
                })

    parsed_all.sort(key=lambda x: x["ts"])
    return parsed_all



def human_readable_seconds(s):
    """Перетворює секунди в людиночитаємий формат (наприклад, '1d 2h 3m 4s')."""
    if s is None or math.isnan(s):
        return "-"
    s = int(round(s))
    d, rem = divmod(s, 86400)
    h, rem = divmod(rem, 3600)
    m, sec = divmod(rem, 60)
    parts = []
    if d: parts.append(f"{d}d")
    if h: parts.append(f"{h}h")
    if m: parts.append(f"{m}m")
    if sec: parts.append(f"{sec}s")
    return " ".join(parts) if parts else "0s"

# ---------- Аналіз ----------
def analyze(messages):
    """Аналізує повідомлення та повертає статистику."""
    valid_msgs = [m for m in messages if m.get("from") or m.get("from_id")]
    users = sorted({str(m["from"] or m["from_id"]) for m in valid_msgs})

    inactivity_threshold = INACTIVITY_THRESHOLD_HOURS * 3600
    starter_counts = Counter()
    starters = []
    prev_ts = None
    for m in valid_msgs:
        if prev_ts is None or (m["ts"] - prev_ts) > inactivity_threshold:
            starter_counts[str(m["from"] or m["from_id"])] += 1
            starters.append((str(m["from"] or m["from_id"]), m["ts"]))
        prev_ts = m["ts"]

    response_times_by_user = defaultdict(list)
    for i in range(1, len(valid_msgs)):
        prev = valid_msgs[i-1]
        cur = valid_msgs[i]
        if (prev.get("from") or prev.get("from_id")) and (cur.get("from") or cur.get("from_id")):
            if str(prev["from"] or prev["from_id"]) != str(cur["from"] or cur["from_id"]):
                delta = cur["ts"] - prev["ts"]
                if MIN_RESPONSE_SECONDS <= delta <= MAX_RESPONSE_SECONDS:
                    response_times_by_user[str(cur["from"] or cur["from_id"])].append(delta)

    lengths_by_user = defaultdict(list)
    words_by_user = defaultdict(list)
    media_counts = Counter()
    weekday_hour = defaultdict(lambda: np.zeros((7, 24), dtype=int))
    timeline = defaultdict(int)
    streaks = defaultdict(int)
    current_streak_user = None
    current_streak_len = 0

    texts_by_user = defaultdict(list)  # новий словник для збереження текстів
    
    for m in valid_msgs:
        user = str(m["from"] or m["from_id"])
        text = m["text"] if isinstance(m["text"], str) else " ".join([t for t in m["text"] if isinstance(t, str)])
        lengths_by_user[user].append(len(text))
        texts_by_user[user].append(text) 

        tokens = re.findall(r"\b\w+\b", text.lower())
        words = [w for w in tokens if w not in STOPWORDS]
        words_by_user[user].extend(words)
        if m["is_media"]:
            media_counts[user] += 1

        dt = datetime.fromtimestamp(m["ts"])
        weekday_hour[user][dt.weekday(), dt.hour] += 1
        timeline[(dt.year, dt.month)] += 1

        if current_streak_user == user:
            current_streak_len += 1
        else:
            if current_streak_user:
                streaks[current_streak_user] = max(streaks[current_streak_user], current_streak_len)
            current_streak_user = user
            current_streak_len = 1
    if current_streak_user:
        streaks[current_streak_user] = max(streaks[current_streak_user], current_streak_len)

    stats = []
    for u in users:
        times = response_times_by_user.get(u, [])
        mean = float(np.mean(times)) if times else float("nan")
        median = float(np.median(times)) if times else float("nan")
        p90 = float(np.percentile(times, 90)) if times else float("nan")
        stats.append({
            "user": u,
            "starter_count": starter_counts.get(u, 0),
            "responses_count": len(times),
            "mean_response": mean,
            "median_response": median,
            "p90_response": p90,
            "avg_length_chars": np.mean(lengths_by_user[u]) if lengths_by_user[u] else 0,
            "avg_length_words": np.mean([
                len(re.findall(r"\b\w+\b", txt)) for txt in texts_by_user[u]
            ]) if texts_by_user[u] else 0,
            "media_percent": (media_counts[u] / len(lengths_by_user[u]) * 100) if lengths_by_user[u] else 0,
            "max_streak": streaks.get(u, 0)
        })

    return {
        "users": users,
        "stats_df": pd.DataFrame(stats),
        "words_by_user": words_by_user,
        "weekday_hour": weekday_hour,
        "timeline": timeline
    }

# ---------- Візуалізація ----------
def visualize(results):
    """Створює та зберігає графіки за результатами аналізу."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = results["stats_df"]
    
    # Використання human_readable_seconds для форматування часу відповіді
    for col in ["mean_response", "median_response", "p90_response"]:
        df[col + "_hr"] = df[col].apply(human_readable_seconds)

    df.to_csv(os.path.join(OUTPUT_DIR, "summary.csv"), index=False)


    plt.figure(figsize=(8,4))
    plt.bar(df["user"], df["starter_count"])
    plt.title("Кількість стартів")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "starters.png"))
    plt.close()

    plt.figure(figsize=(8,4))
    plt.bar(df["user"], df["mean_response"]/60)
    plt.title("Середній час відповіді (хв)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "mean_response.png"))
    plt.close()

    for u in results["users"]:
        mat = results["weekday_hour"][u]
        plt.figure(figsize=(8,4))
        sns.heatmap(mat, cmap="Blues", xticklabels=range(24),
                    yticklabels=["Пн","Вт","Ср","Чт","Пт","Сб","Нд"])
        plt.title(f"Активність по годинах ({u})")
        plt.xlabel("Година")
        plt.ylabel("День тижня")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"heatmap_{safe_filename(u)}.png"))
        plt.close()

    for u, words in results["words_by_user"].items():
        if not words:
            continue
        wc = WordCloud(width=800, height=400, background_color="white").generate(" ".join(words))
        wc.to_file(os.path.join(OUTPUT_DIR, f"wordcloud_{safe_filename(u)}.png"))

    months = sorted(results["timeline"])
    vals = [results["timeline"][m] for m in months]
    labels = [f"{y}-{m:02d}" for y, m in months]
    plt.figure(figsize=(10,4))
    plt.plot(labels, vals, marker="o")
    plt.xticks(rotation=45)
    plt.title("Таймлайн активності")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "timeline.png"))
    plt.close()

# ---------- main ----------
def main():
    """Головна функція для виконання скрипту."""
    files = ["message_1.json", "message_2.json", "result.json"]  # можна додати більше
    msgs = load_messages(files)
    print(f"Loaded {len(msgs)} messages")
    res = analyze(msgs)
    print(res["stats_df"])
    visualize(res)
    print(f"Saved graphs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()