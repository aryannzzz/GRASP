#!/bin/bash
# Monitor training progress for both models

echo "============================================================"
echo "ACT TRAINING PROGRESS MONITOR"
echo "============================================================"
echo "Time: $(date)"
echo ""

# Check if processes are running
echo "=== PROCESS STATUS ==="
STANDARD_PID=$(ps aux | grep "train_act_proper.py.*standard.*diverse" | grep -v grep | head -1 | awk '{print $2}')
MODIFIED_PID=$(ps aux | grep "train_act_proper.py.*modified.*diverse" | grep -v grep | head -1 | awk '{print $2}')

if [ -n "$STANDARD_PID" ]; then
    echo "✅ Standard ACT training: RUNNING (PID: $STANDARD_PID)"
else
    echo "⚠️  Standard ACT training: NOT RUNNING"
fi

if [ -n "$MODIFIED_PID" ]; then
    echo "✅ Modified ACT training: RUNNING (PID: $MODIFIED_PID)"
else
    echo "⚠️  Modified ACT training: NOT RUNNING"
fi

echo ""
echo "=== STANDARD ACT PROGRESS ==="
if [ -f "logs/train_standard_diverse.log" ]; then
    # Get last epoch line
    LAST_EPOCH=$(grep -E "^Epoch [0-9]+/500" logs/train_standard_diverse.log | tail -1)
    if [ -n "$LAST_EPOCH" ]; then
        echo "$LAST_EPOCH"
    else
        echo "Training started, waiting for first epoch..."
    fi
    
    # Get latest validation loss
    VAL_LOSS=$(grep "Val   - L1:" logs/train_standard_diverse.log | tail -1)
    if [ -n "$VAL_LOSS" ]; then
        echo "$VAL_LOSS"
    fi
else
    echo "Log file not found"
fi

echo ""
echo "=== MODIFIED ACT PROGRESS ==="
if [ -f "logs/train_modified_diverse.log" ]; then
    # Get last epoch line
    LAST_EPOCH=$(grep -E "^Epoch [0-9]+/500" logs/train_modified_diverse.log | tail -1)
    if [ -n "$LAST_EPOCH" ]; then
        echo "$LAST_EPOCH"
    else
        echo "Training started, waiting for first epoch..."
    fi
    
    # Get latest validation loss
    VAL_LOSS=$(grep "Val   - L1:" logs/train_modified_diverse.log | tail -1)
    if [ -n "$VAL_LOSS" ]; then
        echo "$VAL_LOSS"
    fi
else
    echo "Log file not found"
fi

echo ""
echo "=== CHECKPOINTS ==="
if [ -d "checkpoints_proper/standard" ]; then
    STANDARD_CKPT=$(ls -t checkpoints_proper/standard/*.pth 2>/dev/null | head -1)
    if [ -n "$STANDARD_CKPT" ]; then
        echo "Standard ACT: $(basename $STANDARD_CKPT) ($(ls -lh $STANDARD_CKPT | awk '{print $5}'))"
    else
        echo "Standard ACT: No checkpoint yet"
    fi
fi

if [ -d "checkpoints_proper/modified" ]; then
    MODIFIED_CKPT=$(ls -t checkpoints_proper/modified/*.pth 2>/dev/null | head -1)
    if [ -n "$MODIFIED_CKPT" ]; then
        echo "Modified ACT: $(basename $MODIFIED_CKPT) ($(ls -lh $MODIFIED_CKPT | awk '{print $5}'))"
    else
        echo "Modified ACT: No checkpoint yet"
    fi
fi

echo ""
echo "============================================================"
echo "To see full logs:"
echo "  Standard: tail -f logs/train_standard_diverse.log"
echo "  Modified: tail -f logs/train_modified_diverse.log"
echo "============================================================"
