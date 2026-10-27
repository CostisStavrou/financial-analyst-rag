#!/usr/bin/env bash
# Embed the next batch of chunks with gemini-embedding-001 on a free-tier key.
#
# The free tier allows about 1,000 embedding requests a day (one per chunk), reset at
# midnight Pacific time. Run this once a day: it resumes where the last run stopped,
# embeds the filings that FinanceBench questions use first, paces itself under the
# per-minute limits, and stops cleanly when the daily quota is used up.
# Needs Docker Desktop running; it does not start Docker itself.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/logs
log=data/logs/embed_daily.log

if ! docker info >/dev/null 2>&1; then
    echo "Docker is not running. Start Docker Desktop, then run this again." >&2
    exit 1
fi
docker compose up -d db >/dev/null 2>&1

echo "=== $(date '+%Y-%m-%d %H:%M %Z')" >>"$log"
EMBED_RPM="${EMBED_RPM:-90}" EMBED_TPM="${EMBED_TPM:-28000}" \
    .venv/bin/python -m finrag index --financebench-first 2>&1 | tee -a "$log"
status=${PIPESTATUS[0]}

if [ "$status" -ne 0 ] && ! tail -n 3 "$log" | grep -q "daily quota reached"; then
    echo "Embedding failed (exit $status); see $log" >&2
    exit "$status"
fi
.venv/bin/python -m finrag stats | tail -n 1 | tee -a "$log"
