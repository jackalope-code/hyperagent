"""HyperAgents — Generate Loop
==============================
Entry point for the DGM-HyperAgents evolution algorithm.

Usage
-----
    python generate_loop.py --domain math_qa
    python generate_loop.py --domain math_qa --max_generations 5 --eval_samples 10
    python generate_loop.py --domain math_qa --resume_from outputs/run_math_qa_20260101_120000

Algorithm overview
------------------
1.  Initialise git repo + archive.
2.  Evaluate the initial task_agent.py; record score.
3.  For each generation:
    a. Reset repo to root commit.
    b. Apply the parent generation's lineage diffs.
    c. Commit that state as the "base" for this generation.
    d. Run MetaAgent — it explores and modifies any file it likes.
    e. Capture git diff (MetaAgent's changes relative to base commit).
    f. Evaluate the modified task_agent.py in a subprocess.
    g. Store score + patch file in the archive.
    h. Select the next parent (score-proportional by default).
4.  Reset repo to root commit on exit.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import datetime

import openai

from dotenv import load_dotenv

load_dotenv()

from meta_agent import MetaAgent
from utils.archive import (
    add_node,
    get_lineage_patches,
    load_archive,
    select_parent,
    update_node,
)
from utils.git_utils import (
    apply_patch,
    get_current_diff,
    get_head_commit,
    init_repo,
    reset_to_commit,
    stage_and_commit,
)

# ---------------------------------------------------------------------------
# Domain registry
# ---------------------------------------------------------------------------
_DOMAIN_REGISTRY: dict[str, tuple[str, str]] = {
    "math_qa": ("domains.math_qa.domain", "MathQADomain"),
    "word_problems": ("domains.word_problems.domain", "WordProblemsDomain"),
    "competition_math": ("domains.competition_math.domain", "CompetitionMathDomain"),
}


def _load_domain(domain_name: str):
    import importlib

    if domain_name not in _DOMAIN_REGISTRY:
        raise ValueError(
            f"Unknown domain {domain_name!r}. Available: {list(_DOMAIN_REGISTRY)}"
        )
    module_path, cls_name = _DOMAIN_REGISTRY[domain_name]
    mod = importlib.import_module(module_path)
    return getattr(mod, cls_name)()


# ---------------------------------------------------------------------------
# Isolated evaluation via subprocess
# ---------------------------------------------------------------------------

def _evaluate_task_agent(
    domain_name: str,
    repo_path: str,
    n_samples: int,
    task_model: str = None,
) -> dict:
    """Evaluate task_agent.py in a fresh subprocess to pick up any code changes."""
    module_path, cls_name = _DOMAIN_REGISTRY[domain_name]

    eval_code = f"""
import sys, json, os
sys.path.insert(0, {repo_path!r})
import importlib

mod = importlib.import_module({module_path!r})
DomainClass = getattr(mod, {cls_name!r})

from task_agent import TaskAgent

domain = DomainClass()
agent = TaskAgent(model={task_model!r}, log=lambda m: None)
results = domain.evaluate(agent, n={n_samples!r})
print(json.dumps(results))
"""

    result = subprocess.run(
        [sys.executable, "-c", eval_code],
        capture_output=True,
        text=True,
        timeout=600,
        cwd=repo_path,
    )

    if result.returncode != 0:
        return {
            "score": 0.0,
            "results": [],
            "n": 0,
            "error": result.stderr[-2000:],
        }

    try:
        output_lines = result.stdout.strip().split("\n")
        return json.loads(output_lines[-1])
    except Exception:
        return {
            "score": 0.0,
            "results": [],
            "n": 0,
            "error": f"Could not parse output:\n{result.stdout[-1000:]}",
        }


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def generate_loop(
    domain_name: str,
    max_generations: int = 10,
    eval_samples: int = -1,
    parent_selection: str = "score_prop",
    output_dir: str = None,
    repo_path: str = ".",
    resume_from: str = None,
    task_model: str = None,
    meta_model: str = None,
) -> str:
    repo_path = os.path.abspath(repo_path)

    # Output directory
    if resume_from:
        output_dir = os.path.abspath(resume_from)
    else:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = output_dir or os.path.join(
            repo_path, "outputs", f"run_{domain_name}_{run_id}"
        )
    os.makedirs(output_dir, exist_ok=True)

    # Logging
    log_path = os.path.join(output_dir, "generate_loop.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    log = logging.getLogger("hyperagents")

    log.info(f"Output directory : {output_dir}")
    log.info(f"Domain           : {domain_name}")
    log.info(f"Max generations  : {max_generations}")
    log.info(f"Task model       : {task_model or 'default'}")
    log.info(f"Meta model       : {meta_model or 'default'}")
    root_commit = get_head_commit(repo_path)
    log.info(f"Root commit      : {root_commit}")

    # Archive
    archive_path = os.path.join(output_dir, "archive.jsonl")
    archive = load_archive(archive_path) if resume_from else []

    if not archive:
        log.info("Evaluating initial task agent …")
        initial_results = _evaluate_task_agent(domain_name, repo_path, eval_samples, task_model)
        log.info(f"Initial score: {initial_results['score']:.4f}  (n={initial_results['n']})")
        if "error" in initial_results:
            log.warning(f"Eval error: {initial_results['error']}")

        gen_dir = os.path.join(output_dir, "gen_initial")
        os.makedirs(gen_dir, exist_ok=True)
        with open(os.path.join(gen_dir, "scores.json"), "w", encoding="utf-8") as f:
            json.dump(initial_results, f, indent=2)

        archive = add_node(
            archive, archive_path, "initial", score=initial_results["score"]
        )

    parent_id = select_parent(archive, method=parent_selection)
    log.info(f"Starting parent  : {parent_id}")

    # Determine starting generation index
    gen_ids = [n["id"] for n in archive if isinstance(n["id"], int)]
    start_gen = (max(gen_ids) + 1) if gen_ids else 1

    try:
        for gen_id in range(start_gen, max_generations + 1):
            log.info(f"\n{'=' * 60}")
            log.info(f"Generation {gen_id}  |  parent={parent_id}")
            log.info(f"{'=' * 60}")

            gen_dir = os.path.join(output_dir, f"gen_{gen_id}")
            os.makedirs(gen_dir, exist_ok=True)

            # ── Build the parent state ────────────────────────────────────
            reset_to_commit(repo_path, root_commit)
            lineage = get_lineage_patches(archive, parent_id)
            log.info(f"Applying {len(lineage)} lineage patch(es) …")
            for patch_path in lineage:
                ok = apply_patch(repo_path, patch_path)
                if not ok:
                    log.warning(f"  Failed to apply {patch_path}")

            # Commit the parent state so we can diff against it
            base_commit = stage_and_commit(repo_path, f"gen_{gen_id}_base")
            log.info(f"Base commit: {base_commit}")

            # ── Run MetaAgent ────────────────────────────────────────────
            parent_eval_dir = os.path.join(output_dir, f"gen_{parent_id}")
            log.info("Running MetaAgent …")
            meta = MetaAgent(model=meta_model, log=log.info)
            try:
                meta.forward(
                    repo_path=repo_path,
                    eval_path=parent_eval_dir,
                    iterations_left=max_generations - gen_id,
                    domain_name=domain_name,
                )
            except openai.RateLimitError as exc:
                log.warning(f"RateLimitError from MetaAgent: {exc}; skipping generation.")
                results = {"score": 0.0, "results": [], "n": 0}
                with open(os.path.join(gen_dir, "scores.json"), "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2)
                archive = add_node(archive, archive_path, gen_id, parent_id=parent_id, score=0.0)
                parent_id = select_parent(archive, method=parent_selection)
                continue
            log.info("MetaAgent finished.")

            # ── Capture diff ─────────────────────────────────────────────
            diff = get_current_diff(repo_path, base_commit)
            patch_file = os.path.join(gen_dir, "model_patch.diff")
            with open(patch_file, "w", encoding="utf-8") as f:
                f.write(diff)
            log.info(f"Diff: {len(diff)} chars → {patch_file}")

            # ── Evaluate ─────────────────────────────────────────────────
            if diff.strip():
                log.info("Evaluating modified task agent …")
                results = _evaluate_task_agent(domain_name, repo_path, eval_samples, task_model)
                log.info(
                    f"Gen {gen_id} score: {results['score']:.4f}  (n={results['n']})"
                )
                if "error" in results:
                    log.warning(f"Eval error: {results['error']}")
            else:
                log.info("No changes from MetaAgent; score=0.")
                results = {"score": 0.0, "results": [], "n": 0}

            with open(os.path.join(gen_dir, "scores.json"), "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)

            # ── Update archive ────────────────────────────────────────────
            archive = add_node(
                archive,
                archive_path,
                gen_id,
                parent_id=parent_id,
                score=results["score"],
                patch_file=patch_file if diff.strip() else None,
            )

            # ── Select next parent ────────────────────────────────────────
            parent_id = select_parent(archive, method=parent_selection)
            log.info(f"Next parent: {parent_id}")

    finally:
        reset_to_commit(repo_path, root_commit)
        log.info(f"\nLoop complete. Results in: {output_dir}")

    return output_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="HyperAgents — self-referential self-improving agent loop"
    )
    parser.add_argument(
        "--domain",
        required=True,
        choices=list(_DOMAIN_REGISTRY),
        help="Evaluation domain",
    )
    parser.add_argument(
        "--max_generations",
        type=int,
        default=10,
        help="Number of evolution generations (default: 10)",
    )
    parser.add_argument(
        "--eval_samples",
        type=int,
        default=-1,
        help="Samples to evaluate per generation (-1 = all, default: -1)",
    )
    parser.add_argument(
        "--parent_selection",
        default="score_prop",
        choices=["random", "latest", "best", "score_prop"],
        help="Parent selection strategy (default: score_prop)",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory (default: outputs/run_<domain>_<timestamp>)",
    )
    parser.add_argument(
        "--resume_from",
        default=None,
        help="Resume an existing run from this output directory",
    )
    parser.add_argument(
        "--task_model",
        default=None,
        help="Model for TaskAgent evaluation (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--meta_model",
        default=None,
        help="Model for MetaAgent improvement (default: gpt-4o)",
    )

    args = parser.parse_args()
    generate_loop(
        domain_name=args.domain,
        max_generations=args.max_generations,
        eval_samples=args.eval_samples,
        parent_selection=args.parent_selection,
        output_dir=args.output_dir,
        resume_from=args.resume_from,
        task_model=args.task_model,
        meta_model=args.meta_model,
    )
