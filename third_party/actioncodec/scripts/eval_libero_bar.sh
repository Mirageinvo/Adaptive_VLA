#!/bin/bash

# =============================================================================
# LIBERO Evaluation Script for Blockwise Autoregressive Model
# =============================================================================

# Default parameters (can be overridden via command line)
CKPT_DIR="${CKPT_DIR:-ZibinDong/SmolVLM2-2.2B-ActionCodec-BAR-LIBERO}"
DEVICE="${DEVICE:-cuda:0}"
IMG_AUG="${IMG_AUG:-True}"
TILED_IMG="${TILED_IMG:-True}"
ENSEMBLE="${ENSEMBLE:-True}"
HORIZON="${HORIZON:-4}"
OUTPUT_DIR="${OUTPUT_DIR:-./evaluation/bar/}"
CFG_PATH="${CFG_PATH:-config/eval/bar.yaml}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --ckpt_dir)
            CKPT_DIR="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --img_aug)
            IMG_AUG="$2"
            shift 2
            ;;
        --tiled_img)
            TILED_IMG="$2"
            shift 2
            ;;
        --ensemble)
            ENSEMBLE="$2"
            shift 2
            ;;
        --horizon)
            HORIZON="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --cfg_path)
            CFG_PATH="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=============================================="
echo "LIBERO Evaluation Configuration"
echo "=============================================="
echo "Checkpoint: $CKPT_DIR"
echo "Device: $DEVICE"
echo "Image Aug: $IMG_AUG"
echo "Tiled Image: $TILED_IMG"
echo "Ensemble: $ENSEMBLE"
echo "Horizon: $HORIZON"
echo "Output Dir: $OUTPUT_DIR"
echo "CFG Path: $CFG_PATH"
echo "=============================================="
echo ""


# Goal suite pos_offset
POS_OFFSET_GOAL=(4 4 4 4 4 4 4 4 3 3)

# Spatial suite pos_offset
POS_OFFSET_SPATIAL=(4 4 4 4 3 3 3 3 4 3)

# Object suite pos_offset
POS_OFFSET_OBJECT=(4 4 4 4 4 4 4 4 4 3)

# Long (10) suite pos_offset
POS_OFFSET_LONG=(4 3 4 4 3 4 3 3 4 3)

# Function to run evaluation for a task suite
run_suite() {
    local suite=$1
    local display_name=$2
    local -n pos_offsets=$3

    echo ""
    echo "=============================================="
    echo "Running $display_name suite..."
    echo "=============================================="

    for task_id in {0..9}; do
        echo "Running task $task_id with pos_offset=${pos_offsets[$task_id]}..."
        python scripts/eval_libero.py \
            --task_id $task_id \
            --ckpt_dir "$CKPT_DIR" \
            --device "$DEVICE" \
            --img_aug $IMG_AUG \
            --tiled_img $TILED_IMG \
            --task_suite "$suite" \
            --ensemble $ENSEMBLE \
            --horizon $HORIZON \
            --pos_offset ${pos_offsets[$task_id]} \
            --output_dir "$OUTPUT_DIR" \
            --cfg_path "$CFG_PATH"
    done
}

# Run all task suites
run_suite "goal" "Goal" POS_OFFSET_GOAL
run_suite "spatial" "Spatial" POS_OFFSET_SPATIAL
run_suite "object" "Object" POS_OFFSET_OBJECT
run_suite "10" "Long" POS_OFFSET_LONG

echo ""
echo ""

# =============================================================================
# Results Summary - Using Python for clean formatting
# =============================================================================

python3 << EOF
import os
import json

def print_results(output_dir):
    suites = [
        ("goal", "Goal"),
        ("spatial", "Spatial"),
        ("object", "Object"),
        ("long", "Long"),
    ]

    # Table width
    W = 102

    print("╔" + "═" * W + "╗")
    print("║" + "LIBERO EVALUATION RESULTS".center(W) + "║")
    print("╠" + "═" * W + "╣")

    overall_success = 0
    overall_total = 0
    suite_stats = []

    for suite_dir, display_name in suites:
        print("║" + " " * W + "║")
        print("║" + f"  【{display_name}】".ljust(W) + "║")
        print("║" + " " * W + "║")
        print("║" + f"  {'Task':<6}  {'Description':<66}  {'Success':>8}  {'Rate':>8}  " + "║")
        print("║" + "  " + "-" * 6 + "  " + "-" * 66 + "  " + "-" * 8 + "  " + "-" * 8 + "  " + "║")

        suite_success = 0
        suite_total = 0

        for task_id in range(10):
            json_file = os.path.join(output_dir, suite_dir, str(task_id), "results.json")
            if os.path.exists(json_file):
                with open(json_file) as f:
                    data = json.load(f)
                desc = data['task_description'][:64]
                success_count = data['success_count']
                total_count = data['total_tests']
                rate = data['success_rate'] * 100

                print(f"║  {task_id:<6}  {desc:<66}  {success_count:>3}/{total_count:<4}  {rate:>6.1f}%  ║")

                suite_success += success_count
                suite_total += total_count
            else:
                print(f"║  {task_id:<6}  {'(results not found)':<66}  {'-':>8}  {'-':>8}  ║")

        print("║" + " " * W + "║")

        if suite_total > 0:
            suite_rate = suite_success / suite_total * 100
            summary = f"{display_name} Average: {suite_success}/{suite_total} ({suite_rate:.1f}%)"
            print("║" + f"  {summary}".ljust(W) + "║")
            suite_stats.append((display_name, suite_success, suite_total, suite_rate))

        overall_success += suite_success
        overall_total += suite_total

        print("╠" + "═" * W + "╣")

    # Overall Summary
    print("║" + " " * W + "║")
    print("║" + "  【Overall Summary】".ljust(W) + "║")
    print("║" + " " * W + "║")

    for name, succ, total, rate in suite_stats:
        line = f"{name:>12}: {succ:>4} / {total:<4} ({rate:>5.1f}%)"
        print("║" + f"  {line}".ljust(W) + "║")

    print("║" + " " * W + "║")

    if overall_total > 0:
        overall_rate = overall_success / overall_total * 100
        print("║" + "  " + "─" * 40 + " " * (W - 42) + "║")
        line = f"{'OVERALL':>12}: {overall_success:>4} / {overall_total:<4} ({overall_rate:>5.1f}%)"
        print("║" + f"  {line}".ljust(W) + "║")

    print("║" + " " * W + "║")
    print("╚" + "═" * W + "╝")

print_results("$OUTPUT_DIR")
EOF

echo ""
echo "Evaluation completed! Results saved in $OUTPUT_DIR"
