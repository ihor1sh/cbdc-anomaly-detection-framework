#!/usr/bin/env bash
# ============================================================================
# Pre-flight check: verifies the environment is ready to run the experiment.
# Run this BEFORE `docker compose up`.
# ============================================================================

set -e

echo "============================================================"
echo "CBDC ML Framework - pre-flight check"
echo "============================================================"

# 1. Docker available?
if ! command -v docker >/dev/null 2>&1; then
    echo "[FAIL] docker is not installed or not on PATH"
    exit 1
fi
echo "[OK]   docker found: $(docker --version)"

# 2. docker compose plugin?
if ! docker compose version >/dev/null 2>&1; then
    echo "[FAIL] 'docker compose' subcommand not available - install Docker Compose v2 plugin"
    exit 1
fi
echo "[OK]   docker compose found: $(docker compose version --short)"

# 3. Elliptic++ data present?
DATA_DIR="$(dirname "$0")/../data/elliptic_pp"
REQUIRED_FILES=(
    "txs_features.csv"
    "txs_classes.csv"
    "txs_edgelist.csv"
    "wallets_features.csv"
    "wallets_classes.csv"
)

missing=0
for f in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$DATA_DIR/$f" ]; then
        echo "[MISSING] $DATA_DIR/$f"
        missing=$((missing + 1))
    else
        size_mb=$(du -m "$DATA_DIR/$f" | awk '{print $1}')
        echo "[OK]      $f (${size_mb} MB)"
    fi
done

if [ "$missing" -gt 0 ]; then
    echo ""
    echo "----------------------------------------------------------"
    echo "$missing file(s) missing from $DATA_DIR"
    echo ""
    echo "Download Elliptic++ from:"
    echo "  https://github.com/git-disl/EllipticPlusPlus"
    echo ""
    echo "Place the CSV files into:"
    echo "  $(realpath "$DATA_DIR")"
    echo "----------------------------------------------------------"
    exit 1
fi

# 4. Disk space (need ~5 GB free for data + images)
free_gb=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
if [ "$free_gb" -lt 5 ]; then
    echo "[WARN] Only ${free_gb} GB free on this disk - 5+ GB recommended"
else
    echo "[OK]   ${free_gb} GB free disk space"
fi

# 5. RAM (recommend 8 GB+)
total_ram_gb=$(grep MemTotal /proc/meminfo 2>/dev/null | awk '{print int($2/1024/1024)}')
if [ -n "$total_ram_gb" ] && [ "$total_ram_gb" -lt 8 ]; then
    echo "[WARN] Only ${total_ram_gb} GB RAM - 8+ GB recommended for full experiment"
elif [ -n "$total_ram_gb" ]; then
    echo "[OK]   ${total_ram_gb} GB RAM"
fi

echo ""
echo "============================================================"
echo "All checks passed - ready to run:"
echo "  docker compose up -d"
echo "============================================================"
