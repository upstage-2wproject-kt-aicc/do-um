"""CLI entrypoint for workflow execution with dummy JSON input."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from src.workflow.graph import execute_workflow_json, execute_workflow_nlu_json


def main() -> None:
    """Runs workflow execution from a JSON file path."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/nlu_dummy_input.json",
        help="Path to workflow input JSON file.",
    )
    parser.add_argument(
        "--input-type",
        choices=("workflow", "nlu"),
        default="nlu",
        help="Input JSON schema type.",
    )
    parser.add_argument(
        "--output",
        default="data/workflow_output_generated.json",
        help="Path to save workflow output JSON file.",
    )
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Print full JSON payload to stdout.",
    )
    args = parser.parse_args()
    if args.input_type == "workflow":
        outputs = asyncio.run(execute_workflow_json(args.input))
    else:
        outputs = asyncio.run(execute_workflow_nlu_json(args.input))
    payload = [item.model_dump(mode="json") for item in outputs]
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    if args.print_json:
        print(text)
    else:
        print(f"Processed items: {len(payload)}")
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
