# Sourced by the side-experiment jobs (13_*, 14_*). Sets CHAT_FLAG to "" or
# "--chat" for $MODEL: the paper's Qwen agents always use the raw prompt; any
# other model is probed (raw first, then chat template) and the job stops if
# neither mode yields parseable verdicts. The choice is cached per model.
TAG=$(echo "$MODEL" | tr '/' '_')
MODE_FILE="runs/side/mode_${TAG}.txt"
mkdir -p runs/side
if [[ "$MODEL" == Qwen/* ]]; then
    CHAT_FLAG=""
elif [ -f "$MODE_FILE" ]; then
    CHAT_FLAG=$(cat "$MODE_FILE")
else
    echo "=== probe: $MODEL, raw prompt ==="
    if python3 scripts/probe_side_agent.py --model-id "$MODEL"; then
        CHAT_FLAG=""
    else
        echo "=== probe: $MODEL, chat template ==="
        if python3 scripts/probe_side_agent.py --model-id "$MODEL" --chat; then
            CHAT_FLAG="--chat"
        else
            echo "ERROR: $MODEL gives unparseable verdicts in both modes" >&2
            exit 1
        fi
    fi
    echo -n "$CHAT_FLAG" > "$MODE_FILE"
fi
CTAG="${TAG}${CHAT_FLAG:+_chat}"
echo "mode for $MODEL: '${CHAT_FLAG:-raw}'"
