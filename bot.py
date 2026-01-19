import os
import sqlite3
import socket
import time
import numpy as np
import faiss
import asyncio
import sys
import tempfile
import subprocess

from telegram import (
    Update,
    BotCommand,
    BotCommandScopeDefault,
    BotCommandScopeChat,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
)
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters
)
from telegram.helpers import escape_markdown
from telegram.constants import ParseMode
from telegram.request import HTTPXRequest
from telethon import TelegramClient
from telethon.errors import FloodWaitError
from telethon.sessions import StringSession
from telethon.tl.types import InputMessagesFilterPhotos

from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
)
from tensorflow.keras.preprocessing import image

# ================= NETWORK SAFETY =================
socket.setdefaulttimeout(30)
faiss.omp_set_num_threads(2)

FAISS_INDEX_PATH = "faiss.index"
FAISS_RESET_MARKER = ".faiss_reset_done"
MIN_TRAIN_SIZE = 1000

USER_COMMANDS = [
    BotCommand("start", "Start the bot"),
    BotCommand("groups", "Manage saved groups"),
    BotCommand("start_scan", "Scan groups"),
    BotCommand("stop_scan", "Stop scan"),
    BotCommand("status", "Show scan status"),
    BotCommand("top", "Set number of similar images"),
    BotCommand("whoami", "Show your access role"),
    BotCommand("chat_id", "Get chat ID"),
    BotCommand("help", "Show help"),
]

ADMIN_COMMANDS = USER_COMMANDS + [
    BotCommand("allow", "Allow a user"),
    BotCommand("disallow", "Remove a user"),
    BotCommand("list_allowed", "List allowed users"),
    BotCommand("backup", "Download DB backup"),
    BotCommand("import", "Import database"),
    BotCommand("reset", "Reset database"),
    BotCommand("reset_session", "Reset Telegram Session"),
    BotCommand("reload_config", "Reload config"),
    BotCommand("restart", "Restart bot"),
    BotCommand("rebuild_index", "Rebuild index"),
    BotCommand("stats", "Show system statistics"),
]

USER_HELP_TEXT = (
    "<b>📘 Available Commands</b>\n\n"
    "/start — Start the bot\n"
    "/groups — Manage saved groups\n"
    "/groups &lt;chat_id&gt; — Add groups\n"
    "/groups -r — Remove groups\n"
    "/start_scan — Scan images from groups\n"
    "/stop_scan — Stop running scan\n"
    "/status — Show scan status\n"
    "/top — Show top N results\n"
    "/top &lt;n&gt;— Set top N results\n"
    "/whoami — Show your access role\n"
    "/chat_id — Get current chat ID\n"
    "/help — Show all commands\n"
)

ADMIN_HELP_TEXT = (
    USER_HELP_TEXT +
    "\n<b>👑 Admin Commands</b>\n\n"
    "/allow &lt;user_id&gt; — Allow a user\n"
    "/disallow &lt;user_id&gt; — Remove a user\n"
    "/list_allowed — List allowed users\n"
    "/backup — Download DB backup\n"
    "/import — Import database (merge)\n"
    "/import -f — Import database (force)\n"
    "/reset — Reset database\n"
    "/reset_session — Reset Telegram session\n"
    "/reload_config — Reload configuration\n"
    "/restart — Restart bot\n"
    "/rebuild_index — Rebuild index\n"
    "/stats — Show system statistics\n"
)


ACCESS_DENIED_MSG = "⛔ You are not authorized to use this bot!\nPlease contact administrator (sathishsathi7780@gmail.com)."

# ================= ADMINS =================
ADMIN_USER = 000000000
allowed_users_cache = set()

# 🧹 Cleanup stale FAISS temp file (from crash)
tmp = FAISS_INDEX_PATH + ".tmp"
if os.path.exists(tmp):
    print("🧹 Removing stale FAISS tmp file")
    os.remove(tmp)

def generate_new_string_session(api_id, api_hash):
    # ✅ Create and bind event loop for this thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    with TelegramClient(StringSession(), api_id, api_hash) as client:
        client.start()   # OTP WILL BE ASKED IN PYCHARM CONSOLE
        session = client.session.save()

    loop.close()
    return session


def update_string_session_in_config(new_session, path="config.txt"):
    global cfg
    cfg["STRING_SESSION"] = new_session

    lines = []
    found = False

    with open(path, "r") as f:
        for line in f:
            if line.startswith("STRING_SESSION="):
                lines.append(f"STRING_SESSION={new_session}\n")
                found = True
            else:
                lines.append(line)

    if not found:
        lines.append(f"\nSTRING_SESSION={new_session}\n")

    with open(path, "w") as f:
        f.writelines(lines)

async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update):
        await safe_reply(update, ACCESS_DENIED_MSG)
        return

    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM image_features")
    total = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM image_features WHERE active=1")
    active = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM image_features WHERE active=0")
    inactive = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM image_refs")
    refs = cur.fetchone()[0]

    rebuild_status = "⚠️ REQUIRED" if needs_rebuild else "✅ Healthy"

    text = (
        "📊 *System Stats*\n\n"
        f"🗂️ Total images: {total}\n"
        f"✅ Active images: {active}\n"
        f"❌ Inactive images: {inactive}\n"
        f"🔗 Total references: {refs}\n"
        f"🧠 FAISS vectors (incl inactive): {index.ntotal}\n"
        f"♻️ Index rebuild: {rebuild_status}\n"
    )

    await update.message.reply_text(
        escape_markdown(text, version=2),
        parse_mode=ParseMode.MARKDOWN_V2
    )


async def reset_session(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update):
        await safe_reply(update, ACCESS_DENIED_MSG)
        return

    await safe_reply(
        update,
        "🔐 Resetting Telegram session...\n\n"
        "⚠️ OTP will be requested in PyCharm console.\n"
        "Please check terminal."
    )

    try:
        api_id = int(cfg["TELEGRAM_API_ID"])
        api_hash = cfg["TELEGRAM_API_HASH"]

        # ✅ RUN TELETHON LOGIN IN A SEPARATE THREAD
        new_session = await asyncio.to_thread(
            generate_new_string_session,
            api_id,
            api_hash
        )

        update_string_session_in_config(new_session)

        await safe_reply(
            update,
            "✅ New session generated and saved.\n"
            "♻️ Please restart the bot"
        )

        await asyncio.sleep(1)

    except Exception as e:
        await safe_reply(update, f"❌ Session reset failed:\n{e}")


def is_allowed_user(update: Update) -> bool:
    return (
            update.effective_user
            and (
                    update.effective_user.id == ADMIN_USER
                    or update.effective_user.id in allowed_users_cache
            )
    )

def load_allowed_users():
    allowed_users_cache.clear()
    cur = conn.cursor()
    cur.execute("SELECT user_id FROM allowed_users")
    rows = cur.fetchall()
    allowed_users_cache.update([row[0] for row in rows])

def is_admin(update: Update) -> bool:
    return (
        update.effective_user
        and update.effective_user.id == ADMIN_USER
    )

async def allow_user(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update):
        await safe_reply(update, ACCESS_DENIED_MSG)
        return

    if not context.args or not context.args[0].isdigit():
        await safe_reply(update, "Usage: /allow <telegram_user_id>")
        return

    uid = int(context.args[0])

    if uid not in allowed_users_cache:
        cur = conn.cursor()
        cur.execute(
            "INSERT OR IGNORE INTO allowed_users VALUES (?)",
            (uid,)
        )
        conn.commit()
        allowed_users_cache.add(uid)

        await safe_reply(update, f"✅ User `{uid}` added to allowed users")
    else:
        await safe_reply(update, f"✅ User `{uid}` already exists in allowed users")


async def disallow_user(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update):
        await safe_reply(update, ACCESS_DENIED_MSG)
        return

    if not context.args or not context.args[0].isdigit():
        await safe_reply(update, "Usage: /disallow <telegram_user_id>")
        return

    uid = int(context.args[0])

    if uid in allowed_users_cache:

        if uid != ADMIN_USER:
            cur = conn.cursor()
            cur.execute(
                "DELETE FROM allowed_users WHERE user_id=?",
                (uid,)
            )
            conn.commit()
            allowed_users_cache.remove(uid)

            await safe_reply(update, f"🗑️ User `{uid}` removed")
    else:
        await safe_reply(update, f"⚠️ User `{uid}` not found")

async def list_allowed(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update):
        await safe_reply(update, ACCESS_DENIED_MSG)
        return

    if not allowed_users_cache:
        await safe_reply(update, "⚠️ No allowed users found")
        return

    lines = ["👥 *Allowed Users*\n"]
    for uid in sorted(allowed_users_cache):
        if uid == ADMIN_USER:
            lines.append(f"• `{uid}`  👑 *ADMIN*")
        else:
            lines.append(f"• `{uid}`")

    await update.message.reply_text(
        escape_markdown("\n".join(lines), version=2),
        parse_mode=ParseMode.MARKDOWN_V2
    )

# 🔥 ONE-TIME FAISS RESET (TEMPORARY FIX)
def reset_index():
    global FAISS_INDEX_PATH, FAISS_RESET_MARKER
    if os.path.exists(FAISS_INDEX_PATH) and not os.path.exists(FAISS_RESET_MARKER):
        print("⚠️ One-time FAISS reset: deleting old faiss.index")
        try:
            os.remove(FAISS_INDEX_PATH)
            with open(FAISS_RESET_MARKER, "w") as f:
                f.write("done")
        except Exception as e:
            print("❌ Failed to delete faiss.index:", e)


def safe_train_ivf(vectors: np.ndarray):
    n = len(vectors)
    if n < 2:
        return False

    effective_nlist = min(index.nlist, n // 2)
    if effective_nlist < 1:
        return False

    orig_nlist = index.nlist
    if effective_nlist != orig_nlist:
        index.nlist = effective_nlist

    index.train(vectors)

    index.nlist = orig_nlist  # restore
    return True


def compute_nlist(vector_count):
    if vector_count < 1000:
        return max(1, vector_count // 2)
    return min(4096, int(np.sqrt(vector_count)))


# ================= CONFIG =================
def load_config(path="config.txt"):
    cfg = {}
    with open(path) as f:
        for line in f:
            if "=" in line:
                k, v = line.strip().split("=", 1)
                cfg[k.strip()] = v.strip()
    return cfg

cfg = load_config()

CURRENT_CONFIG = cfg.copy()
BOT_TOKEN = cfg["BOT_TOKEN"]
API_ID = int(cfg["TELEGRAM_API_ID"])
API_HASH = cfg["TELEGRAM_API_HASH"]
STRING_SESSION = cfg["STRING_SESSION"]

DB_PATH = "features.db"
VECTOR_DIM = 1280
MAX_DB_SIZE = 500 * 1024 * 1024

# ================= GLOBAL STATE =================
index = None
load_cancel_flags = {}   # user_id -> bool
scan_status = {}         # user_id -> dict
user_top_n = {}
scan_tasks = {}   # user_id -> asyncio.Task
scan_progress_msg = {}  # user_id -> Message
faiss_lock = asyncio.Lock()

# ================= REBUILD STATE =================
needs_rebuild = False
rebuild_event = asyncio.Event()

# ================= MODEL =================
model = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")

# ================= DATABASE =================
conn = sqlite3.connect(DB_PATH, check_same_thread=False)

conn.execute("PRAGMA journal_mode=WAL")
conn.execute("PRAGMA synchronous=NORMAL")

cur = conn.cursor()

# ================= ALLOWED USERS =================
cur.execute("""
CREATE TABLE IF NOT EXISTS allowed_users (
    user_id INTEGER PRIMARY KEY
)
""")
conn.commit()

cur.execute("INSERT OR IGNORE INTO allowed_users VALUES (?)", (ADMIN_USER,))
conn.commit()

load_allowed_users()

# cur.execute("DROP TABLE IF EXISTS image_features")
cur.execute("""
CREATE TABLE IF NOT EXISTS image_features (
    image_key TEXT PRIMARY KEY,
    vector BLOB,
    active INTEGER DEFAULT 1
)
""")

cur.execute("""
CREATE INDEX IF NOT EXISTS idx_image_active
ON image_features(active)
""")

# cur.execute("DROP TABLE IF EXISTS image_refs")
cur.execute("""
CREATE TABLE IF NOT EXISTS image_refs (
    image_key TEXT,
    chat_id   TEXT,
    msg_id    INTEGER,
    chat_title TEXT,
    PRIMARY KEY (image_key, chat_id, msg_id)
)
""")

cur.execute("""
CREATE INDEX IF NOT EXISTS idx_image_refs_key
ON image_refs(image_key)
""")


# cur.execute("DROP TABLE IF EXISTS user_groups")
cur.execute("""
CREATE TABLE IF NOT EXISTS user_groups (
    user_id INTEGER,
    chat_id TEXT,
    chat_title TEXT,
    PRIMARY KEY (user_id, chat_id)
)
""")

# cur.execute("DROP TABLE IF EXISTS group_progress")
cur.execute("""
CREATE TABLE IF NOT EXISTS group_progress (
    chat_id TEXT PRIMARY KEY,
    last_msg_id INTEGER
)
""")
conn.commit()

# ================= FAISS =================
def create_faiss_index(vector_count):
    nlist = compute_nlist(vector_count)
    quantizer = faiss.IndexFlatIP(VECTOR_DIM)
    idx = faiss.IndexIVFFlat(quantizer, VECTOR_DIM, nlist, faiss.METRIC_INNER_PRODUCT)
    idx.nprobe = min(16, nlist)
    return idx

image_keys = []

def get_image_links(image_key):
    cur = conn.cursor()
    cur.execute(
        """
        SELECT chat_id, msg_id, chat_title
        FROM image_refs WHERE image_key=?
        ORDER BY msg_id
        """,
        (image_key,)
    )
    return cur.fetchall()

async def get_image_links_live(image_key, client):
    global needs_rebuild
    cur = conn.cursor()
    cur.execute(
        """
        SELECT chat_id, msg_id, chat_title
        FROM image_refs
        WHERE image_key=?
        """,
        (image_key,)
    )

    rows = cur.fetchall()
    valid_links = []

    dirty = False
    for chat_id, msg_id, title in rows:
        try:
            msg = await client.get_messages(int(chat_id), ids=msg_id)
            if msg:
                valid_links.append((chat_id, msg_id, title))
            else:
                raise Exception("Deleted")

        except Exception:
            # ❌ Message deleted → remove ref only
            cur.execute(
                """
                DELETE FROM image_refs
                WHERE image_key=? AND chat_id=? AND msg_id=?
                """,
                (image_key, chat_id, msg_id)
            )
            dirty = True
    if dirty:
        conn.commit()

    # 🔍 If no refs remain → mark inactive
    cur.execute(
        "SELECT COUNT(*) FROM image_refs WHERE image_key=?",
        (image_key,)
    )
    remaining = cur.fetchone()[0]

    if remaining == 0:
        cur.execute(
            """
            UPDATE image_features
            SET active = 0
            WHERE image_key=?
            """,
            (image_key,)
        )
        conn.commit()
        needs_rebuild = True
        rebuild_event.set()

    cur.execute("SELECT COUNT(*) FROM image_features WHERE active=0")
    inactive = cur.fetchone()[0]

    if inactive > max(500, index.ntotal // 3):
        if not needs_rebuild:
            print("⚠️ FAISS marked for rebuild (too many inactive vectors)")
            needs_rebuild = True
            rebuild_event.set()

    return valid_links


def reload_config():
    global cfg, BOT_TOKEN, API_ID, API_HASH, STRING_SESSION

    new_cfg = load_config()

    changed_critical = any(
        new_cfg.get(k) != CURRENT_CONFIG.get(k)
        for k in ("BOT_TOKEN", "TELEGRAM_API_ID", "TELEGRAM_API_HASH", "STRING_SESSION")
    )

    cfg.clear()
    cfg.update(new_cfg)

    CURRENT_CONFIG.clear()
    CURRENT_CONFIG.update(new_cfg)

    return changed_critical

async def reload_new_config(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update):
        await safe_reply(update, ACCESS_DENIED_MSG)
        return

    needs_restart = reload_config()

    if needs_restart:
        await safe_reply(
            update,
            "♻️ Config reloaded\n⚠️ Restart required to apply critical changes"
        )
    else:
        await safe_reply(update, "♻️ Config reloaded successfully")

async def set_top_n(update: Update, context: ContextTypes.DEFAULT_TYPE):

    if not is_allowed_user(update):
        await safe_reply(update, ACCESS_DENIED_MSG)
        return

    uid = update.effective_user.id

    # Show current value
    if not context.args:
        current = user_top_n.get(uid, 5)
        await update.message.reply_text(f"🔍 Current top results count: {current}")
        return

    # Validate input
    try:
        n = int(context.args[0])
        if not (1 <= n <= 20):
            raise ValueError
    except ValueError:
        await update.message.reply_text(
            "⚠️ Invalid value.\n"
            "Usage: /top <number>\n"
            "Allowed range: 1 – 20"
        )
        return

    user_top_n[uid] = n
    await update.message.reply_text(f"✅ Top results set to: {n}")


async def load_vectors():
    global index
    cur = conn.cursor()
    cur.execute(
        "SELECT image_key, vector FROM image_features WHERE active = 1"
    )

    rows = cur.fetchall()

    if not rows:
        return

    if len(rows) < 2:
        print("⚠️ Not enough vectors to train FAISS index")
        return

    vectors = []
    keys = []

    for k, v in rows:
        vec = np.frombuffer(v, dtype=np.float32)
        vectors.append(vec)
        keys.append(k)

    arr = np.array(vectors)
    faiss.normalize_L2(arr)

    MAX_TRAIN = 50000

    index.reset()
    image_keys.clear()
    if not index.is_trained:
        if len(arr) > MAX_TRAIN:
            sample = arr[np.random.choice(len(arr), MAX_TRAIN, replace=False)]
        else:
            sample = arr

        trained = safe_train_ivf(sample)

        if trained:
            index.add(arr)
            image_keys.extend(keys)
        else:
            print("⚠️ Not enough vectors to train IVF on load")
            return
    else:
        index.add(arr)
        image_keys.extend(keys)


# ================= HELPERS =================
def get_user_groups(uid):
    cur = conn.cursor()
    cur.execute(
        "SELECT chat_id FROM user_groups WHERE user_id=?",
        (uid,)
    )

    return [r[0] for r in cur.fetchall()]

def get_last_msg_id(chat_id):

    cur = conn.cursor()

    cur.execute(
        "SELECT last_msg_id FROM group_progress WHERE chat_id=?",
        (str(chat_id),)
    )

    row = cur.fetchone()

    return row[0] if row else None

def update_last_msg_id(chat_id, msg_id):
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO group_progress (chat_id, last_msg_id)
        VALUES (?, ?)
        ON CONFLICT(chat_id)
        DO UPDATE SET last_msg_id=excluded.last_msg_id
        """,
        (str(chat_id), msg_id),
    )

def extract_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    vec = model.predict(x, verbose=0).astype("float32")
    faiss.normalize_L2(vec)   # normalize 2D array
    return vec[0]             # return normalized vector


async def search_similar(img_path, top_n):
    vec = extract_features(img_path)
    async with faiss_lock:
        # search more to compensate inactive skips
        k = min(index.ntotal, max(top_n * 5, 50))
        D, I = index.search(vec.reshape(1, -1), k)
        keys_snapshot = list(image_keys)

    results = []
    cur = conn.cursor()
    seen = set()

    for score, idx in zip(D[0], I[0]):
        if idx < 0 or idx >= len(keys_snapshot):
            continue

        image_key = keys_snapshot[idx]
        if image_key in seen:
            continue
        seen.add(image_key)

        cur.execute(
            "SELECT active FROM image_features WHERE image_key=?",
            (image_key,)
        )
        row = cur.fetchone()

        if not row or row[0] == 0:
            continue  # ❌ skip inactive

        results.append((image_key, float(score)))

    return results

def build_message_link(chat_id, msg_id):

    cid = abs(int(chat_id))
    internal = cid - 1000000000000  # correct internal id
    return f"https://t.me/c/{internal}/{msg_id}"

def format_eta(seconds):
    if seconds < 60:
        return f"{int(seconds)} sec"
    return f"{int(seconds//60)}m {int(seconds%60)}s"

def parse_load_args(args, uid):
    limit = None
    if not args:
        return get_user_groups(uid), None

    if args[-1].isdigit():
        limit = int(args[-1])
        groups = args[:-1]
    else:
        groups = args

    if not groups:
        groups = get_user_groups(uid)

    return groups, limit

async def safe_reply_md(update, text):
    try:
        await update.message.reply_text(
            escape_markdown(text, version=2),
            parse_mode=ParseMode.MARKDOWN_V2
        )
    except Exception as e:
        print("safe_reply_md error:", e)


async def safe_reply(update, text):
    try:
        await update.message.reply_text(text)
    except Exception as e:
        print("Exception caught (300): ",e)

# ================= BACKUP (TEMP ONLY) =================
def create_temp_backup():
    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    tmp = f"features_backup_{ts}.db"
    conn.commit()
    with open(DB_PATH, "rb") as src, open(tmp, "wb") as dst:
        dst.write(src.read())
    return tmp

# ================= COMMANDS =================
async def set_groups(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global STRING_SESSION

    if not is_allowed_user(update):
        await safe_reply(update, ACCESS_DENIED_MSG)
        return

    uid = update.effective_user.id
    args = context.args
    cursor = conn.cursor()
    # ================= SHOW GROUPS =================
    if not args:
        cursor.execute(
            "SELECT chat_title, chat_id FROM user_groups WHERE user_id=?",
            (uid,)
        )
        rows = cursor.fetchall()

        if not rows:
            await safe_reply(update, "📂 No groups saved")
            return

        text = "📂 *Saved groups:*\n\n"

        for title, cid in rows:
            safe_title = escape_markdown(title or "Unknown", version=2)
            safe_cid = escape_markdown(str(cid), version=2)

            text += f"• {safe_title}\n  `{safe_cid}`\n\n"

        await update.message.reply_text(
            text,
            parse_mode=ParseMode.MARKDOWN_V2
        )

        return

    # ================= REMOVE MODE =================
    if args[0] in ("-r", "remove"):
        if len(args) == 1:

            cursor.execute(
                "DELETE FROM user_groups WHERE user_id=?",
                (uid,)
            )
            conn.commit()
            await safe_reply(update, "🗑️ All groups removed")
            return

        removed = 0

        for target in args[1:]:
            # Case 1: chat_id
            if target.lstrip("-").isdigit():
                cursor.execute(
                    "DELETE FROM user_groups WHERE user_id=? AND chat_id=?",
                    (uid, str(target))
                )
                removed += cursor.rowcount

            # Case 2: group title (case-insensitive)
            else:
                cursor.execute(
                    """
                    DELETE FROM user_groups
                    WHERE user_id=? AND LOWER(chat_title)=LOWER(?)
                    """,
                    (uid, target)
                )
                removed += cursor.rowcount

        conn.commit()

        if removed:
            await safe_reply(update, f"🗑️ Removed {removed} group(s)")
        else:
            await safe_reply(update, "⚠️ No matching groups found")

        return

    # ================= ADD MODE =================

    if STRING_SESSION != cfg["STRING_SESSION"]:
        STRING_SESSION = cfg["STRING_SESSION"]

    client = TelegramClient(StringSession(STRING_SESSION), API_ID, API_HASH)
    await client.start()

    added = []

    async with client:
        for chat_id in args:
            try:
                if chat_id.lstrip("-").isdigit():

                    try:
                        entity = await client.get_entity(int(chat_id))
                        title = entity.title
                    except Exception as e:
                        title = f"group_{chat_id}"

                    cursor.execute(
                        "INSERT OR IGNORE INTO user_groups VALUES (?, ?, ?)",
                        (uid, chat_id, title)
                    )
                    conn.commit()

                    if cursor.rowcount:
                        added.append(title)

                else:
                    await safe_reply(update, f"❌ Invalid chat id: {chat_id}")
                    continue

            except Exception as e:
                print(f"Cannot access group: {chat_id}\nReason: {e}")
                await safe_reply(update, f"Cannot access group: {chat_id}\nReason: {e}")

    if added:
        safe_lines = "\n".join(
            f"• {escape_markdown(grp, version=2)}" for grp in added
        )

        await update.message.reply_text(
            "✅ Added groups:\n" + safe_lines,
            parse_mode=ParseMode.MARKDOWN_V2
        )

    else:
        await safe_reply(update, "ℹ️ No new groups added")


async def safe_delete(path, retries=5, delay=0.2):
    for _ in range(retries):
        try:
            os.remove(path)
            return
        except PermissionError:
            await asyncio.sleep(delay)
    # last attempt
    try:
        os.remove(path)
    except Exception as e:
        print("Exception caught (431): ",e)

async def get_chat_id(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat = update.effective_chat

    title = escape_markdown(chat.title or "Private Chat", version=2)
    chat_id = escape_markdown(str(chat.id), version=2)

    await update.message.reply_text(
        f"*Chat title:* {title}\n*Chat ID:* `{chat_id}`",
        parse_mode=ParseMode.MARKDOWN_V2
    )

async def rebuild_index(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update):
        await safe_reply(update, ACCESS_DENIED_MSG)
        return

    if any(s.get("running") for s in scan_status.values()):
        await safe_reply(
            update,
            "♻️ Rebuilding cancelled! Some background processes are still running"
        )
        return

    await safe_reply(
        update,
        "♻️ Rebuilding FAISS index...\n"
        "⏳ Searches will be temporarily paused."
    )

    global needs_rebuild

    start = time.time()

    try:
        async with faiss_lock:   # ✅ REQUIRED
            await load_vectors()

            tmp_index = FAISS_INDEX_PATH + ".tmp"
            faiss.write_index(index, tmp_index)
            os.replace(tmp_index, FAISS_INDEX_PATH)

            if os.path.exists(tmp_index):
                os.remove(tmp_index)

        print(f"♻️ FAISS rebuilt in {time.time() - start:.2f}s")
        needs_rebuild = False
        rebuild_event.clear()
        await safe_reply(update, "✅ FAISS index rebuilt successfully")

    except Exception as e:
        await safe_reply(update, f"❌ Rebuild failed:\n{e}")


async def run_scan(update: Update, context: ContextTypes.DEFAULT_TYPE):

    global STRING_SESSION, index, needs_rebuild

    if not is_allowed_user(update):
        await safe_reply(update, ACCESS_DENIED_MSG)
        return

    uid = update.effective_user.id
    scan_status.setdefault(uid, {})
    scan_status[uid].update({
        "running": True,
        "start": time.time(),
        "processed": 0,
        "added": 0,
        "skipped": 0
    })

    try:
        cur = conn.cursor()
        groups, limit = parse_load_args(context.args, uid)

        if not groups:
            await safe_reply(update, "⚠️ No groups specified or saved")
            scan_status[uid]["running"] = False
            return

        progress = await update.message.reply_text(
            "📥 Scan started...\nProcessed: 0\nAdded: 0\nSkipped: 0"
        )
        scan_progress_msg[uid] = progress

        processed = added = skipped = 0
        last_update = time.time()
        start = time.time()

        if STRING_SESSION != cfg["STRING_SESSION"]:
            STRING_SESSION = cfg["STRING_SESSION"]

        client = TelegramClient(StringSession(STRING_SESSION), API_ID, API_HASH)
        await client.start()

        async with client:
            for group in groups:
                last_seen_msg_id = None
                chat_id = str(group)
                if not chat_id.startswith("-"):
                    continue

                try:
                    last_id = get_last_msg_id(chat_id) or 0
                    async for msg in client.iter_messages(
                            int(chat_id),
                            filter=InputMessagesFilterPhotos,
                            reverse=True,
                            min_id=last_id,
                            limit=limit or 10_000
                    ):
                        # 🛑 Cancel support
                        if load_cancel_flags.get(uid):
                            await progress.edit_text(
                                f"🛑 Scan cancelled\n\n"
                                f"Processed: {processed}\n"
                                f"Added: {added}\n"
                                f"Skipped: {skipped}"
                            )
                            scan_status[uid]["running"] = False
                            return

                        processed += 1
                        scan_status[uid]["processed"] = processed

                        if not msg.photo:
                            continue

                        last_seen_msg_id = msg.id

                        # 🔑 Stable image key (Telethon-safe)
                        photo = msg.photo
                        image_key = f"tg_{photo.id}_{photo.access_hash}"

                        chat = msg.chat
                        chat_title = getattr(chat, "title", "Unknown")

                        cur.execute(
                            """
                            INSERT OR IGNORE INTO image_refs
                            (image_key, chat_id, msg_id, chat_title)
                            VALUES (?, ?, ?, ?)
                            """,
                            (
                                image_key,
                                str(msg.chat_id),
                                msg.id,
                                chat_title
                            )
                        )

                        cur.execute(
                            "SELECT active, vector FROM image_features WHERE image_key=?",
                            (image_key,)
                        )
                        row = cur.fetchone()

                        if row is None:
                            # ⬇️ Download & extract ONLY once per unique image
                            fd, tmp = tempfile.mkstemp(suffix=".jpg")
                            os.close(fd)

                            try:
                                await client.download_media(msg.media, tmp)
                                vec = await asyncio.to_thread(extract_features, tmp)

                                cur.execute(
                                    "INSERT INTO image_features VALUES (?, ?, ?)",
                                    (image_key, vec.tobytes(), 1)
                                )

                                needs_rebuild = True
                                rebuild_event.set()
                                added += 1
                                scan_status[uid]["added"] = added

                            finally:
                                await safe_delete(tmp)

                        elif row[0] == 0:
                            cur.execute(
                                "UPDATE image_features SET active = 1 WHERE image_key=?",
                                (image_key,)
                            )

                            conn.commit()
                            needs_rebuild = True
                            rebuild_event.set()
                            added += 1

                        else:
                            # 🔁 Already active image
                            skipped += 1
                            scan_status[uid]["skipped"] = skipped

                        await asyncio.sleep(0)

                        if processed % 50 == 0:
                            update_last_msg_id(chat_id, msg.id)
                            conn.commit()

                        # 📊 Progress update
                        if time.time() - last_update >= 6:
                            avg = (time.time() - start) / processed
                            eta = avg * 10
                            await progress.edit_text(
                                f"📥 Scanning...\n"
                                f"Processed: {processed}\n"
                                f"Added: {added}\n"
                                f"Skipped: {skipped}\n"
                                f"ETA: ~{format_eta(eta)}"
                            )
                            last_update = time.time()

                    if last_seen_msg_id is not None:
                        update_last_msg_id(chat_id, last_seen_msg_id)
                        conn.commit()

                except FloodWaitError as e:
                    print(f"⏳ FloodWait: sleeping {e.seconds}s")
                    await asyncio.sleep(e.seconds)

                except Exception as e:
                    print(f"[scan-warning] {type(e).__name__}")
                    await asyncio.sleep(2) # small cooldown

            conn.commit()

        scan_status[uid]["running"] = False
        await progress.edit_text(
            f"✅ Scan completed\n"
            f"Processed: {processed}\n"
            f"Added: {added}\n"
            f"Skipped: {skipped}"
        )

    except asyncio.CancelledError:
        scan_status[uid]["running"] = False
        raise

    finally:
        if uid in scan_status:
            scan_status[uid]["running"] = False
            scan_status[uid]["end"] = time.time()
        scan_tasks.pop(uid, None)
        scan_progress_msg.pop(uid, None)
        load_cancel_flags.pop(uid, None)

        if needs_rebuild:
            print("♻️ DB updated, FAISS rebuild pending")

# ================= LOAD / CANCEL =================
async def start_scan(update: Update, context: ContextTypes.DEFAULT_TYPE):

    if not is_allowed_user(update):
        await safe_reply(update, ACCESS_DENIED_MSG)
        return

    uid = update.effective_user.id

    if uid in scan_tasks and not scan_tasks[uid].done():
        await safe_reply(update, "⚠️ Scan already running")
        return

    load_cancel_flags[uid] = False

    groups, limit = parse_load_args(context.args, uid)
    if not groups:
        await safe_reply(update, "⚠️ No groups specified or saved")
        return

    scan_tasks[uid] = asyncio.create_task(run_scan(update, context))


async def stop_scan(update: Update, context: ContextTypes.DEFAULT_TYPE):

    if not is_allowed_user(update):
        await safe_reply(update, ACCESS_DENIED_MSG)
        return

    uid = update.effective_user.id

    if not scan_status.get(uid, {}).get("running"):
        await safe_reply(update, "⚠️ No active scan")
        return

    load_cancel_flags[uid] = True
    progress = scan_progress_msg.get(uid)
    if progress:
        stats = scan_status.get(uid, {})
        processed = stats.get("processed", 0)
        added = stats.get("added", 0)
        skipped = stats.get("skipped", 0)

        try:
            await progress.edit_text(
                f"🛑 Scan stopped by user\n\n"
                f"Processed: {processed}\n"
                f"Added: {added}\n"
                f"Skipped: {skipped}"
            )
        except Exception as e:
            print("Progress edit failed:", e)

    await safe_reply(update, "🛑 Scan cancellation requested")


# ================= STATUS =================
async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):

    if not is_allowed_user(update):
        await safe_reply(update, ACCESS_DENIED_MSG)
        return

    uid = update.effective_user.id
    status = scan_status.get(uid)

    # DB count
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM image_features WHERE active = 1")
    db_count = cur.fetchone()[0]

    text = "📊 *Status*\n\n"

    if db_count < 2:
        text += "\n⚠️ Not enough images to train search index"

    text += f"🗂️ Images in DB: {db_count}\n\n"
    # text += f"🧠 Images in FAISS index: {index.ntotal}\n\n"

    if not status:
        text += "🔍 Scan running: ❌"
        await update.message.reply_text(
            escape_markdown(text, version=2),
            parse_mode=ParseMode.MARKDOWN_V2
        )

        return

    running = status["running"]
    uptime = time.time() - status["start"]

    text += f"🔍 Scan running: {'✅' if running else '❌'}\n"
    text += f"⏱️ Uptime: {format_eta(uptime)}\n"

    await update.message.reply_text(
    escape_markdown(text, version=2),
    parse_mode=ParseMode.MARKDOWN_V2
)


# ================= BACKUP / RESET =================
async def backup_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update):
        await safe_reply(update, ACCESS_DENIED_MSG)
        return

    tmp = create_temp_backup()
    try:
        with open(tmp, "rb") as f:
            await update.message.reply_document(
                document=f,
                filename=os.path.basename(tmp),
                caption="🗄️ Database backup",
            )
    finally:
        os.remove(tmp)

async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update):
        await safe_reply(update, ACCESS_DENIED_MSG)
        return

    keyboard = InlineKeyboardMarkup([
        [
            InlineKeyboardButton("🗑️ Proceed", callback_data="reset_yes"),
            InlineKeyboardButton("❌ Cancel", callback_data="reset_no"),
        ]
    ])
    await update.message.reply_text(
        "⚠️ *DANGEROUS OPERATION*\n\n"
        "This will delete ALL image features.\n"
        "A backup will be sent before deletion.\n\n"
        "Proceed?",
        parse_mode="Markdown",
        reply_markup=keyboard,
    )

async def reset_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()

    if q.data == "reset_no":
        await q.edit_message_text("❌ Reset cancelled.")
        return

    if q.data == "reset_yes":
        tmp = create_temp_backup()
        try:
            with open(tmp, "rb") as f:
                await q.message.reply_document(
                    document=f,
                    filename=os.path.basename(tmp),
                    caption="🗄️ Backup before reset",
                )
        finally:
            os.remove(tmp)

        cur = conn.cursor()
        cur.execute("DELETE FROM image_features")
        cur.execute("DELETE FROM image_refs")
        cur.execute("DELETE FROM group_progress")
        conn.commit()

        reset_index()

        async with faiss_lock:
            index.reset()
            image_keys.clear()

        await q.edit_message_text("✅ Reset completed successfully.")

async def rebuild_watcher():
    global needs_rebuild
    while True:
        await rebuild_event.wait()
        await asyncio.sleep(10)   # debounce
        rebuild_event.clear()

        if not needs_rebuild:
            continue

        if any(s.get("running") for s in scan_status.values()):
            rebuild_event.set()
            continue

        print("🛠️ Background FAISS rebuild started")

        try:
            async with faiss_lock:
                await load_vectors()

                if index.ntotal < 2:
                    print("⚠️ Rebuild skipped: not enough vectors")
                    needs_rebuild = False
                    continue

                tmp_index = FAISS_INDEX_PATH + ".tmp"
                faiss.write_index(index, tmp_index)
                os.replace(tmp_index, FAISS_INDEX_PATH)

                if os.path.exists(tmp_index):
                    os.remove(tmp_index)

            needs_rebuild = False
            print("✅ Background FAISS rebuild completed")

        except Exception as e:
            print("❌ Background rebuild failed:", e)
            rebuild_event.set()

# ================= SEARCH =================
async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed_user(update):
        await safe_reply(update, ACCESS_DENIED_MSG)
        return

    global index
    if index is None or not index.is_trained:
        await safe_reply(update, "⚠️ Search index not ready. Please try again.")
        return

    if needs_rebuild:
        await safe_reply(
            update,
            "⚠️ Search index is updating.\n"
            "Please wait a few minutes or ask admin to run /rebuild_index."
        )
        return

    await safe_reply(update, "✅ Image received. Finding similar images...")

    if not index.is_trained or index.ntotal == 0:
        await safe_reply(update, "⚠️ DB empty. Run /start_scan first.")
        return

    photo = update.message.photo[-1]
    file = await photo.get_file()
    # tmp = f"query_{update.message.message_id}.jpg"
    fd, tmp = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    await file.download_to_drive(tmp)

    top_n = user_top_n.get(update.effective_user.id, 5)
    results = await search_similar(tmp, top_n)

    await safe_delete(tmp)

    text = "🔍 *Top Matches*\n\n"

    client = TelegramClient(StringSession(STRING_SESSION), API_ID, API_HASH)

    async with client:
        # for image_key, score in results:
        #     percent = escape_markdown(f"{score * 100:.2f}", version=2)
        #     text += f"📸 Similarity: {percent}%\n"
        #
        #     links = await get_image_links_live(image_key, client)
        #
        #     for chat_id, msg_id, title in links:
        #         safe_title = escape_markdown(title or "My Group", version=2)
        #         link = escape_markdown(
        #             build_message_link(chat_id, msg_id),
        #             version=2
        #         )
        #         text += f"• {safe_title}\n  🔗 {link}\n"
        #
        #     text += "\n"

        shown = 0
        for image_key, score in results:
            if shown >= top_n:
                break

            links = await get_image_links_live(image_key, client)

            # ❌ Skip images with no remaining references
            if not links:
                continue

            percent = escape_markdown(f"{score * 100:.2f}", version=2)
            text += f"📸 Similarity: {percent}%\n"

            for chat_id, msg_id, title in links:
                safe_title = escape_markdown(title or "My Group", version=2)
                link = escape_markdown(
                    build_message_link(chat_id, msg_id),
                    version=2
                )
                text += f"• {safe_title}\n  🔗 {link}\n"

            text += "\n"
            shown += 1

    await update.message.reply_text(
        text,
        parse_mode=ParseMode.MARKDOWN_V2
    )


async def import_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update):
        await safe_reply(update, ACCESS_DENIED_MSG)
        return

    force = False

    if context.args and context.args[0] == "-f":
        force = True

    context.user_data["import_force"] = force

    mode_text = "FULL REPLACE (-f)" if force else "MERGE MODE"

    await update.message.reply_text(
        f"📥 Import mode: *{mode_text}*\n\n"
        "Please upload the backup `.db` file.\n\n"
        "⚠️ This action cannot be undone.",
        parse_mode="Markdown"
    )

def count_rows_in_db(db_path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM image_features WHERE active = 1")
    total = cur.fetchone()[0]
    conn.close()
    return total

async def handle_config_upload(update: Update, context: ContextTypes.DEFAULT_TYPE):

    if not is_admin(update):
        await safe_reply(update, ACCESS_DENIED_MSG)
        return

    doc = update.message.document
    tmp_path = "config.txt.new"
    await safe_reply(update, "📥 Updating config.txt...")

    try:
        file = await doc.get_file()
        await file.download_to_drive(custom_path=tmp_path)

        # 🔍 Validate format
        new_cfg = load_config(tmp_path)
        required = {"BOT_TOKEN", "TELEGRAM_API_ID", "TELEGRAM_API_HASH", "STRING_SESSION"}
        missing = required - new_cfg.keys()

        if missing:
            os.remove(tmp_path)
            await safe_reply(
                update,
                f"❌ Invalid config.txt\nMissing keys: {', '.join(missing)}"
            )
            return

        # 🔁 Replace config atomically
        os.replace(tmp_path, "config.txt")

        # 🔄 Reload config
        needs_restart = reload_config()

        if needs_restart:
            await safe_reply(
                update,
                "✅ config.txt updated\n"
                "🔁 Critical changes detected\n"
                "♻️ Please restart the bot"
            )
            await asyncio.sleep(1)
        else:
            await safe_reply(
                update,
                "✅ config.txt updated successfully\n"
                "♻️ Reloaded without restart"
            )

    except Exception as e:
        await safe_reply(update, f"❌ Failed to update config.txt\n{e}")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

async def handle_import_file(update: Update, context: ContextTypes.DEFAULT_TYPE):

    if not is_admin(update):
        await safe_reply(update, ACCESS_DENIED_MSG)
        return

    doc = update.message.document
    global needs_rebuild
    if doc.file_size > MAX_DB_SIZE:
        await safe_reply(update, "❌ File too large")
        return

    # ✅ CONFIG FILE HANDLING
    if doc and doc.file_name == "config.txt":
        await handle_config_upload(update, context)
        return

    force = context.user_data.get("import_force", False)

    if not doc or not doc.file_name.endswith(".db"):
        await safe_reply(update, "⚠️ Please upload a valid `.db` backup file")
        return

    tmp_path = f"import_{doc.file_unique_id}.db"
    file = await doc.get_file()

    try:
        await file.download_to_drive(
            custom_path=tmp_path
        )
    except Exception as e:
        await safe_reply(
            update,
            "❌ Download failed (file too large or network issue).\n"
            "Please try again.\n\n"
            f"Reason: {e}"
        )
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return

    progress_msg = await update.message.reply_text(
        "📥 Preparing import...\nProgress: 0%"
    )

    try:
        total_rows = count_rows_in_db(tmp_path)

        if force:
            await progress_msg.edit_text(
                "📥 Importing database (FULL REPLACE)\n"
                f"Total records: {total_rows}\nProgress: 0%"
            )

            # 1️⃣ Clear existing data safely
            cur = conn.cursor()
            cur.execute("DELETE FROM image_features")
            cur.execute("DELETE FROM image_refs")
            cur.execute("DELETE FROM group_progress")
            conn.commit()

            async with faiss_lock:
                index.reset()
                image_keys.clear()

            # 2️⃣ Read from import DB
            src_conn = sqlite3.connect(tmp_path)
            src_cur = src_conn.cursor()
            src_cur.execute("SELECT image_key, vector FROM image_features")

            processed = 0
            last_update = time.time()

            for image_key, vector_blob in src_cur:
                processed += 1

                cur.execute(
                    "INSERT INTO image_features VALUES (?, ?, ?)",
                    (image_key, vector_blob, 1)
                )

                needs_rebuild = True
                rebuild_event.set()

                if time.time() - last_update >= 3:
                    percent = int((processed / total_rows) * 100)
                    await progress_msg.edit_text(
                        "📥 Importing database (FULL REPLACE)\n"
                        f"Processed: {processed}/{total_rows}\n"
                        f"Progress: {percent}%"
                    )
                    last_update = time.time()

            conn.commit()
            src_conn.close()

            await progress_msg.edit_text(
                "✅ Import completed (FULL REPLACE)\n"
                f"📦 Total images: {index.ntotal}"
            )

            return

        # ================= MERGE MODE =================
        src_conn = sqlite3.connect(tmp_path)
        src_cur = src_conn.cursor()
        src_cur.execute("SELECT image_key, vector FROM image_features")

        added = skipped = processed = 0
        last_update = time.time()

        await progress_msg.edit_text(
            "📥 Importing database (MERGE MODE)\n"
            f"Total records: {total_rows}\nProgress: 0%"
        )
        cur = conn.cursor()
        for image_key, vector_blob in src_cur:
            processed += 1

            cur.execute(
                "SELECT 1 FROM image_features WHERE image_key=?",
                (image_key,)
            )
            if cur.fetchone():
                skipped += 1
                continue

            cur.execute(
                "INSERT INTO image_features VALUES (?, ?, ?)",
                (image_key, vector_blob, 1)
            )
            conn.commit()

            needs_rebuild = True
            rebuild_event.set()
            added += 1

            # 🔁 Update progress every 3 seconds
            if time.time() - last_update >= 3:
                percent = int((processed / total_rows) * 100)

                await progress_msg.edit_text(
                    "📥 Importing database (MERGE MODE)\n"
                    f"Processed: {processed} / {total_rows}\n"
                    f"Added: {added}\n"
                    f"Skipped: {skipped}\n"
                    f"Progress: {percent}%"
                )
                last_update = time.time()

        src_conn.close()

        await progress_msg.edit_text(
            "✅ Import completed (MERGE MODE)\n"
            f"Processed: {processed}\n"
            f"Added: {added}\n"
            f"Skipped: {skipped}\n"
            f"📦 Total images: {index.ntotal}"
        )

    except Exception as e:
        await progress_msg.edit_text(f"❌ Import failed:\n{e}")

    finally:
        context.user_data.pop("import_force", None)
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

        needs_rebuild = True
        rebuild_event.set()


# ================= ERROR HANDLER =================
async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    print("ERROR:", context.error)

    if isinstance(update, Update) and update.effective_chat:
        try:
            await context.bot.send_message(
                update.effective_chat.id,
                "⚠️ An internal error occurred."
            )
        except Exception as e:
            print("Exception caught: {}\n{}", sys.exc_info()[0], e)


# ================= SHOW HELP =================
async def show_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        ADMIN_HELP_TEXT if is_admin(update) else USER_HELP_TEXT,
        parse_mode=ParseMode.HTML
    )

# ================= WELCOME =================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id

    # 🚫 Block guests early
    if uid != ADMIN_USER and uid not in allowed_users_cache:
        await safe_reply(
            update,
            ACCESS_DENIED_MSG
        )
        return

    commands_set = context.application.bot_data.setdefault("commands_set", set())

    role = "admin" if uid == ADMIN_USER else "user"
    key = (uid, role)

    if key not in commands_set:
        await context.bot.set_my_commands(
            ADMIN_COMMANDS if role == "admin" else USER_COMMANDS,
            scope=BotCommandScopeChat(chat_id=uid)
        )
        commands_set.add(key)

    await safe_reply(
        update,
        "Welcome to Carving_File_Bot\n\n"
        "--------DEVELOPED BY-------\n"
        "'Keshavarapu Sathish Kumar'\n\n"
        "Try /help for a list of available commands."
    )

async def restart_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update):
        await safe_reply(update, ACCESS_DENIED_MSG)
        return

    await safe_reply(update, "♻️ Restarting bot service...")

    subprocess.Popen(
        ["sudo", "systemctl", "restart", "image-search-bot"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )


async def whoami(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user

    uid = user.id
    safe_name = escape_markdown(user.full_name or "Unknown", version=2)
    safe_uid = escape_markdown(str(uid), version=2)

    if uid == ADMIN_USER:
        role = "👑 *Admin*"
    elif uid in allowed_users_cache:
        role = "✅ *Allowed User*"
    else:
        role = "🚫 *Guest*"

    text = (
        f"👤 *Who Am I*\n\n"
        f"• Name: {safe_name}\n"
        f"• User ID: `{safe_uid}`\n"
        f"• Role: {role}"
    )

    await update.message.reply_text(
        text,
        parse_mode=ParseMode.MARKDOWN_V2
    )


async def setup_commands(application):
    # Default commands → everyone
    await application.bot.set_my_commands(
        USER_COMMANDS,
        scope=BotCommandScopeDefault()
    )

    # Admin-specific commands
    await application.bot.set_my_commands(
        ADMIN_COMMANDS,
        scope=BotCommandScopeChat(chat_id=ADMIN_USER)
    )

    global index
    cur.execute("SELECT COUNT(*) FROM image_features WHERE active = 1")
    vector_count = cur.fetchone()[0]

    if vector_count < 2:
        print("⚠️ Not enough images to train FAISS index")

    expected_nlist = compute_nlist(vector_count)

    if os.path.exists(FAISS_INDEX_PATH):
        try:
            loaded = faiss.read_index(FAISS_INDEX_PATH)

            if loaded.nlist != expected_nlist:
                raise ValueError("nlist mismatch")

            loaded.nprobe = min(16, loaded.nlist)
            index = loaded
            print("✅ FAISS index loaded from disk")

        except Exception as e:
            print("⚠️ FAISS incompatible, rebuilding from DB:", e)
            os.remove(FAISS_INDEX_PATH)
            index = create_faiss_index(vector_count)
            async with faiss_lock:
                await load_vectors()

    else:
        # 🔥 THIS WAS MISSING
        index = create_faiss_index(vector_count)
        async with faiss_lock:
            await load_vectors()

    application.create_task(rebuild_watcher())


# ================= MAIN =================
if __name__ == "__main__":

    request = HTTPXRequest(
        connect_timeout=60,
        read_timeout=300,  # ⬅️ VERY IMPORTANT
        write_timeout=300,
        pool_timeout=60
    )

    app = (
        ApplicationBuilder()
        .token(BOT_TOKEN)
        .request(request)
        .build()
    )

    app.post_init = setup_commands

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("groups", set_groups))
    app.add_handler(CommandHandler("start_scan", start_scan))
    app.add_handler(CommandHandler("stop_scan", stop_scan))
    app.add_handler(CommandHandler("status", status_command))
    app.add_handler(CommandHandler("backup", backup_command))
    app.add_handler(CommandHandler("reset", reset_command))
    app.add_handler(CommandHandler("import", import_command))
    app.add_handler(CommandHandler("top", set_top_n))
    app.add_handler(CommandHandler("reset_session", reset_session))
    app.add_handler(CommandHandler("reload_config", reload_new_config))
    app.add_handler(CommandHandler("chat_id", get_chat_id))
    app.add_handler(CommandHandler("help", show_help))
    app.add_handler(CommandHandler("allow", allow_user))
    app.add_handler(CommandHandler("disallow", disallow_user))
    app.add_handler(CommandHandler("list_allowed", list_allowed))
    app.add_handler(CommandHandler("whoami", whoami))
    app.add_handler(CommandHandler("restart", restart_command))
    app.add_handler(CommandHandler("rebuild_index", rebuild_index))
    app.add_handler(CommandHandler("stats", stats_command))
    app.add_handler(CallbackQueryHandler(reset_callback, pattern="^reset_"))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_import_file))
    app.add_handler(MessageHandler(filters.PHOTO, handle_image))
    app.add_error_handler(error_handler)

    print("🤖 Bot running (FULL FEATURED FINAL VERSION)...")
    app.run_polling(drop_pending_updates=True)
