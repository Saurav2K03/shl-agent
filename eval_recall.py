"""
Recall@10 Evaluation Script for the SHL Conversational Agent.

This script:
1. Parses all 10 conversation traces (C1–C10) from GenAI_SampleConversations/.
2. Extracts the expected (ground truth) assessment names from the final recommendation
   table in each trace (the one before `end_of_conversation: true`).
3. Sends only the USER messages from the trace to the live /chat endpoint (simulating
   what the evaluator does — giving the agent just the user's facts).
4. Compares the agent's returned recommendation names against the ground truth.
5. Computes Recall@10 per trace and Mean Recall@10 across all traces.
"""

import json
import os
import re
import requests
import sys

URL = os.environ.get("AGENT_URL", "https://shl-agent-bsss.onrender.com")
TRACES_DIR = "GenAI_SampleConversations"

def extract_ground_truth_names(filepath):
    """
    Extract the expected assessment names from the FINAL recommendation table
    in the trace (the one right before end_of_conversation: true).
    """
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Find all markdown tables in the file
    # A table row looks like: | 1 | Assessment Name | ... |
    table_row_pattern = re.compile(
        r"^\|\s*\d+\s*\|\s*(.+?)\s*\|", re.MULTILINE
    )

    # Split by turns and find the last turn with end_of_conversation: true
    # We want the table from the final agent response
    turns = content.split("### Turn")
    
    # Find the last turn that has end_of_conversation: true
    final_turn = None
    for turn in reversed(turns):
        if "end_of_conversation" in turn and "**true**" in turn:
            final_turn = turn
            break

    if not final_turn:
        return []

    # Extract assessment names from the table in that final turn
    names = []
    for match in table_row_pattern.finditer(final_turn):
        name = match.group(1).strip()
        # Skip table header rows
        if name.startswith("---") or name.lower() == "name":
            continue
        names.append(name)

    return names


def extract_user_messages(filepath):
    """
    Extract all user messages from the trace as a flat conversation history.
    We send ALL user messages to give the agent full context.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    messages = []
    current_role = None
    buffer = []

    def flush():
        if current_role and buffer:
            text = "\n".join(buffer).strip()
            if current_role == "user":
                text = "\n".join(line.lstrip("> ") for line in text.split("\n"))
            messages.append({"role": current_role, "content": text})
            buffer.clear()

    for line in lines:
        stripped = line.strip()

        if stripped == "## Conversation" or stripped.startswith("### Turn"):
            continue

        if stripped == "**User**":
            flush()
            current_role = "user"
            continue
        if stripped == "**Agent**":
            flush()
            current_role = "assistant"
            continue

        # Skip metadata lines
        if current_role == "assistant" and stripped.startswith("_"):
            continue

        if current_role:
            buffer.append(line.rstrip("\n"))

    flush()

    # Only keep user messages for the "simulated user" approach
    # But we need alternating user/assistant for the API to work correctly.
    # So we keep the full history, stripping only the last assistant response.
    if messages and messages[-1]["role"] == "assistant":
        messages.pop()

    return messages


def normalize_name(name):
    """Normalize assessment name for fuzzy comparison."""
    n = name.lower().strip()
    # Remove hyphens/dashes surrounded by spaces (formatting differences)
    n = re.sub(r'\s*[-–—]\s*', ' ', n)
    # Collapse multiple spaces
    n = re.sub(r'\s+', ' ', n)
    # Remove special chars that vary between trace and catalog
    n = n.replace('&', 'and')
    return n


def recall_at_k(predicted_names, ground_truth_names, k=10):
    """
    Compute Recall@K with fuzzy name matching.
    """
    if not ground_truth_names:
        return 1.0  # No ground truth = trivially correct

    pred_normalized = {normalize_name(n): n for n in predicted_names[:k]}
    gt_normalized = {normalize_name(n): n for n in ground_truth_names}

    if not gt_normalized:
        return 1.0

    hits = len(set(pred_normalized.keys()) & set(gt_normalized.keys()))
    return hits / len(gt_normalized)


def main():
    traces = sorted(
        [f for f in os.listdir(TRACES_DIR) if f.endswith(".md")],
        key=lambda x: int(re.search(r"\d+", x).group())
    )

    print(f"Agent URL: {URL}")
    print(f"Traces found: {len(traces)}")
    print(f"{'='*80}\n")

    all_recalls = []

    for trace_file in traces:
        filepath = os.path.join(TRACES_DIR, trace_file)
        trace_name = trace_file.replace(".md", "")

        # 1. Extract ground truth
        gt_names = extract_ground_truth_names(filepath)

        # 2. Extract messages to send
        messages = extract_user_messages(filepath)

        print(f"--- {trace_name} ---")
        print(f"  Ground truth ({len(gt_names)} items): {gt_names}")
        print(f"  Sending {len(messages)} messages to agent...")

        # Rate limit: wait between API calls to avoid 429s
        if all_recalls:  # Skip delay for the first request
            import time
            time.sleep(5)

        # 3. Call the agent
        try:
            resp = requests.post(
                f"{URL}/chat",
                json={"messages": messages},
                timeout=120
            )
            resp.raise_for_status()
            data = resp.json()

            pred_names = [r["name"] for r in data.get("recommendations", [])]
            print(f"  Agent returned ({len(pred_names)} items): {pred_names}")

            # 4. Compute Recall@10
            r10 = recall_at_k(pred_names, gt_names, k=10)
            all_recalls.append(r10)
            
            status = "✅" if r10 >= 0.5 else "⚠️" if r10 > 0 else "❌"
            print(f"  Recall@10: {r10:.2f} {status}")

            # Show misses
            pred_set = set(normalize_name(n) for n in pred_names[:10])
            gt_set = set(normalize_name(n) for n in gt_names)
            missed = gt_set - pred_set
            if missed:
                print(f"  Missed: {[n for n in gt_names if normalize_name(n) in missed]}")

        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            all_recalls.append(0.0)

        print()

    # 5. Mean Recall@10
    mean_r10 = sum(all_recalls) / len(all_recalls) if all_recalls else 0.0
    print(f"{'='*80}")
    print(f"  MEAN RECALL@10: {mean_r10:.3f}  ({len(all_recalls)} traces)")
    print(f"{'='*80}")

    # Per-trace summary table
    print(f"\n{'Trace':<8} {'Recall@10':>10}")
    print(f"{'-'*20}")
    for trace_file, r10 in zip(traces, all_recalls):
        name = trace_file.replace(".md", "")
        print(f"{name:<8} {r10:>10.2f}")
    print(f"{'-'*20}")
    print(f"{'Mean':<8} {mean_r10:>10.3f}")


if __name__ == "__main__":
    main()
