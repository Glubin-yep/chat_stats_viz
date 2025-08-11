#!/usr/bin/env python3
# Розширений аналіз чату з Telegram та Instagram з підтримкою emoji
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
import math
import regex
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Шрифт з підтримкою emoji
matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans", "Segoe UI Emoji"]
matplotlib.rcParams["font.family"] = ["DejaVu Sans", "Segoe UI Emoji"]


# ---------- Налаштування ----------
FILES = ["message_1.json", "message_2.json", "result.json"]  # можна додати більше
OUTPUT_DIR = "output"
INACTIVITY_THRESHOLD_HOURS = 6.0
MAX_RESPONSE_SECONDS = 60 * 60 * 24 * 7
MIN_RESPONSE_SECONDS = 1

STOPWORDS = set(
    """
і в на що з до це за як але чи а та ж або ну ось вже від бо
""".split()
)


# ---------- Допоміжні функції ----------
def safe_filename(name):
    # Створює безпечне ім'я файлу, видаляючи неприпустимі символи.
    return re.sub(r"[^0-9A-Za-zА-Яа-яЁёІіЇїЄє _-]", "_", name)


def fix_instagram_text(s):
    # Виправляє текст з Instagram, який зламаний через неправильну кодування.
    if not isinstance(s, str):
        return ""
    try:
        return s.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return s


def human_readable_seconds(s):
    # Перетворює секунди в людиночитаємий формат.
    if s is None or math.isnan(s):
        return "-"
    s = int(round(s))
    d, rem = divmod(s, 86400)
    h, rem = divmod(rem, 3600)
    m, sec = divmod(rem, 60)
    parts = []
    if d:
        parts.append(f"{d}d")
    if h:
        parts.append(f"{h}h")
    if m:
        parts.append(f"{m}m")
    if sec:
        parts.append(f"{sec}s")
    return " ".join(parts) if parts else "0s"


def tokenize_with_emoji(text):
    """Розбиває текст на слова та emoji, фільтруючи сміття."""
    emoji_pattern = r"\p{Extended_Pictographic}"
    word_pattern = r"[a-zA-Zа-яА-ЯёЁіІїЇєЄ]{2,}"  # слова довжиною >= 2
    pattern = f"{emoji_pattern}|{word_pattern}"
    tokens = regex.findall(pattern, text)
    return [t for t in tokens if t.strip()]


# ---------- Завантаження ----------
def load_messages(paths):
    # Завантажує та об'єднує повідомлення з кількох файлів JSON.
    parsed_all = []

    for path in paths:
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Формат Instagram
        if "participants" in data and "messages" in data:
            for m in data.get("messages", []):
                parsed_all.append(
                    {
                        "id": None,
                        "from": fix_instagram_text(m.get("sender_name")),
                        "from_id": fix_instagram_text(m.get("sender_name")),
                        "ts": int(m["timestamp_ms"] / 1000),
                        "text": fix_instagram_text(m.get("content", "")),
                        "is_media": any(
                            k in m for k in ("photos", "videos", "audio_files", "share")
                        ),
                    }
                )
        # Формат Telegram
        else:
            msgs = data.get("messages", [])
            for m in msgs:
                if "date_unixtime" in m and m["date_unixtime"]:
                    ts = int(m["date_unixtime"])
                else:
                    ts = int(
                        datetime.fromisoformat(m["date"])
                        .replace(tzinfo=timezone.utc)
                        .timestamp()
                    )
                text = m.get("text", "")
                if not isinstance(text, str):
                    text = " ".join([t for t in text if isinstance(t, str)])
                parsed_all.append(
                    {
                        "id": m.get("id"),
                        "from": m.get("from"),
                        "from_id": m.get("from_id"),
                        "ts": ts,
                        "text": text,
                        "is_media": any(
                            k in m
                            for k in (
                                "photo",
                                "file",
                                "media_type",
                                "video",
                                "voice_message",
                            )
                        ),
                    }
                )

    parsed_all.sort(key=lambda x: x["ts"])
    return parsed_all


# ---------- Аналіз ----------
def analyze(messages):
    valid_msgs = [m for m in messages if m.get("from") or m.get("from_id")]
    users = sorted({str(m["from"] or m["from_id"]) for m in valid_msgs})

    inactivity_threshold = INACTIVITY_THRESHOLD_HOURS * 3600
    starter_counts = Counter()
    prev_ts = None
    for m in valid_msgs:
        if prev_ts is None or (m["ts"] - prev_ts) > inactivity_threshold:
            starter_counts[str(m["from"] or m["from_id"])] += 1
        prev_ts = m["ts"]

    response_times_by_user = defaultdict(list)
    for i in range(1, len(valid_msgs)):
        prev = valid_msgs[i - 1]
        cur = valid_msgs[i]
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
    texts_by_user = defaultdict(list)

    for m in valid_msgs:
        user = str(m["from"] or m["from_id"])
        text = m["text"]
        lengths_by_user[user].append(len(text))
        texts_by_user[user].append(text)

        tokens = tokenize_with_emoji(text.lower())
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
                streaks[current_streak_user] = max(
                    streaks[current_streak_user], current_streak_len
                )
            current_streak_user = user
            current_streak_len = 1
    if current_streak_user:
        streaks[current_streak_user] = max(
            streaks[current_streak_user], current_streak_len
        )

    stats = []
    for u in users:
        times = response_times_by_user.get(u, [])
        mean = float(np.mean(times)) if times else float("nan")
        median = float(np.median(times)) if times else float("nan")
        p90 = float(np.percentile(times, 90)) if times else float("nan")
        stats.append(
            {
                "user": u,
                "starter_count": starter_counts.get(u, 0),
                "responses_count": len(times),
                "mean_response": mean,
                "median_response": median,
                "p90_response": p90,
                "avg_length_chars": (
                    np.mean(lengths_by_user[u]) if lengths_by_user[u] else 0
                ),
                "avg_length_words": (
                    np.mean([len(tokenize_with_emoji(txt)) for txt in texts_by_user[u]])
                    if texts_by_user[u]
                    else 0
                ),
                "media_percent": (
                    (media_counts[u] / len(lengths_by_user[u]) * 100)
                    if lengths_by_user[u]
                    else 0
                ),
                "max_streak": streaks.get(u, 0),
            }
        )

    return {
        "users": users,
        "stats_df": pd.DataFrame(stats),
        "words_by_user": words_by_user,
        "weekday_hour": weekday_hour,
        "timeline": timeline,
    }


# ---------- Візуалізація ----------
def visualize(results):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = results["stats_df"]

    for col in ["mean_response", "median_response", "p90_response"]:
        df[col + "_hr"] = df[col].apply(human_readable_seconds)

    df.to_csv(os.path.join(OUTPUT_DIR, "summary.csv"), index=False)

    plt.figure(figsize=(8, 4))
    plt.bar(df["user"], df["starter_count"])
    plt.title("Кількість стартів(ініціативність)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "starters.png"))
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.bar(df["user"], df["mean_response"] / 60)
    plt.title("Середній час відповіді (хв)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "mean_response.png"))
    plt.close()

    for u in results["users"]:
        mat = results["weekday_hour"][u]
        plt.figure(figsize=(8, 4))
        sns.heatmap(
            mat,
            cmap="Blues",
            xticklabels=range(24),
            yticklabels=["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Нд"],
        )
        plt.title(f"Активність по годинах ({u})")
        plt.xlabel("Година")
        plt.ylabel("День тижня")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"heatmap_{safe_filename(u)}.png"))
        plt.close()

    for u, words in results["words_by_user"].items():
        if not words:
            continue
        most_common = Counter(words).most_common(20)

        # Вивід у консоль
        print(f"Top words for {u}: " + ", ".join(f"{w} ({c})" for w, c in most_common))

        # Створення DataFrame для таблиці
        df = pd.DataFrame(most_common, columns=["Слова", "Кількість"])

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.axis("off")

        # Створюємо таблицю
        table = ax.table(
            cellText=df.values,
            colLabels=df.columns,
            cellLoc="center",
            loc="center",
            colColours=["#4F81BD", "#4F81BD", "#4F81BD"],  # колір заголовка (синій)
            colWidths=[0.6, 0.3],  # пропорції колонок
        )

        ax.set_title(
            f"Загальна кількість слів {words.__len__()} для {u}", fontweight="bold"
        )

        # Форматування шрифтів і кольорів
        table.auto_set_font_size(True)

        # Стилізуємо заголовок (перший рядок)
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_text_props(weight="bold", color="white")
                cell.set_facecolor("#4F81BD")
        else:
            # Альтернатива для рядків (змінюємо колір фону по черзі)
            if row % 2 == 0:
                cell.set_facecolor("#4F81BD")  # світло-сірий
            else:
                cell.set_facecolor("white")
                cell.set_edgecolor("#CCCCCC")  # світло-сіра межа
                cell.set_linewidth(0.5)

        plt.tight_layout()

        # Збереження у файл PNG
        plt.savefig(
            os.path.join(OUTPUT_DIR, f"table_{safe_filename(u)}.png"),
            bbox_inches="tight",
            dpi=150,
        )
        plt.close(fig)

    months = sorted(results["timeline"])
    vals = [results["timeline"][m] for m in months]
    labels = [f"{y}-{m:02d}" for y, m in months]

    plt.figure(figsize=(12, 6))
    plt.plot(labels, vals, marker="o", linewidth=2, markersize=8, color="#1f77b4")

    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)

    plt.grid(axis="y", linestyle="--", alpha=0.7)  # сітка по осі Y

    plt.title("Таймлайн активності", fontsize=16, weight="bold")
    plt.ylabel("Кількість", fontsize=14)
    plt.xlabel("Місяць", fontsize=14)

    # Додаємо підписи над точками
    for x, y in zip(labels, vals):
        plt.text(x, y + max(vals) * 0.02, str(y), ha="center", fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "timeline.png"))
    plt.close()


# ---------- main ----------
def main():
    msgs = load_messages(FILES)
    print(f"Loaded {len(msgs)} messages")
    res = analyze(msgs)
    print(res["stats_df"])
    visualize(res)
    print(f"Saved graphs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
