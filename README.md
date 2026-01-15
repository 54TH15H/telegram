🖼️ Telegram Image Similarity Search Bot (FAISS + MobileNet)

A production-ready Telegram bot that scans images from Telegram groups, extracts deep learning features, stores them efficiently, and allows reverse image search using cosine similarity powered by FAISS.

Designed for large image collections, group scanning, admin access control, and high performance.

🚀 Features
🔍 Image Similarity Search

Upload an image → get top-N similar images

Uses MobileNetV2 (ImageNet) embeddings

Similarity powered by FAISS IVF index

📥 Group Image Scanning

Scan Telegram groups for images

Incremental scanning using last message tracking

Supports large groups (10k+ images)

FloodWait-safe with automatic backoff

🧠 Smart Indexing

FAISS IndexIVFFlat with dynamic nlist

L2-normalized vectors for cosine similarity

Persistent index stored on disk

Auto rebuild if index becomes incompatible

🗄️ Persistent Storage

SQLite database with WAL mode

Stores:

Image vectors

Image references (group, message link)

User groups

Scan progress

Supports backup & restore

👑 Role-Based Access Control

Admin

Allowed users

Guests are blocked

Admin-only critical operations

⚙️ Production Ready

Async & non-blocking

FAISS thread safety with locks

Safe temp file handling

Config reload without restart

Systemd restart support

🧩 Tech Stack
Component	Technology
Language	Python 3.9+
Bot API	python-telegram-bot v20+
Telegram Client	Telethon
Deep Learning	TensorFlow + MobileNetV2
Vector Search	FAISS
Database	SQLite (WAL mode)
Deployment	systemd
📂 Project Structure
.
├── bot.py                  # Main bot code
├── config.txt              # Runtime configuration
├── features.db             # SQLite database
├── faiss.index             # FAISS index file
├── README.md

⚙️ Configuration (config.txt)
BOT_TOKEN=YOUR_BOT_TOKEN
TELEGRAM_API_ID=123456
TELEGRAM_API_HASH=abcdef1234567890
STRING_SESSION=YOUR_TELETHON_SESSION


⚠️ Never commit config.txt to public repositories.

🔐 Access Control

ADMIN_USER is hardcoded in the source

Admin can:

Allow / disallow users

Reset database

Import / backup DB

Reset Telegram session

Restart bot service

🤖 Bot Commands
👤 User Commands
/start            Start the bot
/help             Show help
/groups           View saved groups
/groups <id>      Add groups
/groups -r        Remove groups
/start_scan       Start scanning images
/stop_scan        Stop scan
/status           Scan status
/top <n>          Set top-N results
/chat_id          Get chat ID
/whoami           Show your role

👑 Admin Commands
/allow <user_id>        Allow user
/disallow <user_id>     Remove user
/list_allowed           List users
/backup                 Download DB
/import                 Import DB (merge)
/import -f              Import DB (force)
/reset                  Reset DB
/reset_session          Reset Telethon session
/reload_config          Reload config
/restart                Restart bot service

🖼️ How Image Search Works

Images are downloaded from Telegram groups

Features extracted using MobileNetV2

Vectors normalized and stored

FAISS IVF index used for fast similarity search

Results returned with:

Similarity score

Direct Telegram message links

🧪 FAISS Index Strategy

Uses Inner Product (cosine similarity)

Dynamic nlist calculation:

Small DB → small clusters

Large DB → √N clusters

Index auto-rebuild if incompatible

🔄 Database Import Modes
Merge Mode (default)

Keeps existing data

Skips duplicate images

Force Mode (/import -f)

Deletes existing data

Rebuilds FAISS index from scratch

🛠️ Deployment (systemd example)
[Unit]
Description=Telegram Image Search Bot
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/bot
ExecStart=/usr/bin/python3 bot.py
Restart=always

[Install]
WantedBy=multi-user.target

🔒 Security Notes

Admin-only destructive operations

Telegram session regeneration supported

SQLite WAL mode for crash safety

Temporary files cleaned safely

OTP required only during session reset

📈 Performance Tips

Increase MIN_TRAIN_SIZE for large datasets

SSD recommended for FAISS index

Avoid running multiple scans concurrently

Use /status to monitor long scans

👨‍💻 Author

Keshavarapu Sathish Kumar
📧 Email: sathishsathi7780@gmail.com

📜 License

This project is intended for private / internal use.
