#!/usr/bin/env python3
"""
Upgrade evaluation system to use exploration-dependent questions (V2).

This script:
1. Backs up current evaluation/tasks.py
2. Creates a new evaluation entry point that uses V2 questions
3. Modifies experiments/runner.py to use trajectory-enhanced ground truth
4. Provides rollback capability

Usage:
    python scripts/upgrade_to_exploration_eval_v2.py --apply
    python scripts/upgrade_to_exploration_eval_v2.py --rollback
"""

import argparse
import shutil
from pathlib import Path
from datetime import datetime


def backup_file(filepath: Path) -> Path:
    """Create timestamped backup of file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = filepath.with_suffix(f".backup_{timestamp}{filepath.suffix}")
    shutil.copy(filepath, backup_path)
    print(f"✅ Backed up {filepath} → {backup_path}")
    return backup_path


def apply_upgrade():
    """Apply the upgrade to exploration-dependent evaluation"""

    print("="*70)
    print("UPGRADING TO EXPLORATION-DEPENDENT EVALUATION (V2)")
    print("="*70)
    print()

    # 1. Backup current tasks.py
    tasks_path = Path("evaluation/tasks.py")
    if tasks_path.exists():
        backup_file(tasks_path)

    # 2. Create new __init__.py that routes to V2 questions
    init_path = Path("evaluation/__init__.py")
    if not init_path.exists() or init_path.read_text().strip() == "":
        # Create new init
        init_content = """# evaluation/__init__.py
# Use V2 exploration-dependent questions by default

from evaluation.tasks_exploration_v2 import (
    get_test_queries_v2 as get_test_queries,
    get_query_statistics_v2 as get_query_statistics,
)
from evaluation.trajectory_extraction import enhance_ground_truth_with_trajectory

__all__ = [
    'get_test_queries',
    'get_query_statistics',
    'enhance_ground_truth_with_trajectory',
]
"""
        init_path.write_text(init_content)
        print(f"✅ Created {init_path}")
    else:
        backup_file(init_path)
        print(f"⚠️  evaluation/__init__.py already exists, backed up but not modified")
        print("   You may need to manually update imports")

    # 3. Modify runner.py to use enhanced ground truth
    runner_path = Path("experiments/runner.py")
    if runner_path.exists():
        backup_file(runner_path)

        runner_code = runner_path.read_text()

        # Add import at top
        import_line = "from evaluation.trajectory_extraction import enhance_ground_truth_with_trajectory\n"
        if import_line.strip() not in runner_code:
            # Find the imports section and add our import
            import_section_end = runner_code.find("\n\nclass")
            if import_section_end != -1:
                runner_code = (
                    runner_code[:import_section_end] +
                    "\n" + import_line +
                    runner_code[import_section_end:]
                )

        # Modify the _evaluate_agent method to enhance ground truth
        old_eval_start = "        # Evaluation (ground truth used ONLY here)"
        new_eval_code = """        # Evaluation (ground truth used ONLY here)
        # UPGRADE V2: Enhance ground truth with trajectory data from steps
        ground_truth_base = env.get_ground_truth()
        ground_truth = enhance_ground_truth_with_trajectory(
            steps=[step.to_dict() for step in steps],
            environment_ground_truth=ground_truth_base,
            environment_name=env.__class__.__name__
        )
"""

        if old_eval_start in runner_code and "enhance_ground_truth_with_trajectory" not in runner_code:
            # Replace the old evaluation section
            old_eval_code = """        # Evaluation (ground truth used ONLY here)
        ground_truth = env.get_ground_truth()
"""
            runner_code = runner_code.replace(old_eval_code, new_eval_code)

            runner_path.write_text(runner_code)
            print(f"✅ Modified {runner_path} to use enhanced ground truth")
        else:
            print(f"⚠️  {runner_path} already modified or structure changed")
            print("   Manual review required")

    print()
    print("="*70)
    print("UPGRADE COMPLETE")
    print("="*70)
    print()
    print("Next steps:")
    print("1. Run verification test: python scripts/run_observer_verification.py")
    print("2. Check that Observer scores <40% on new questions")
    print("3. If successful, run full study with new evaluation")
    print()
    print("To rollback: python scripts/upgrade_to_exploration_eval_v2.py --rollback")
    print()


def rollback_upgrade():
    """Rollback to original evaluation system"""

    print("="*70)
    print("ROLLING BACK TO ORIGINAL EVALUATION")
    print("="*70)
    print()

    # Find most recent backups
    for path in [Path("evaluation/tasks.py"), Path("experiments/runner.py"), Path("evaluation/__init__.py")]:
        backup_dir = path.parent
        backups = sorted(backup_dir.glob(f"{path.stem}.backup_*{path.suffix}"), reverse=True)

        if backups:
            latest_backup = backups[0]
            shutil.copy(latest_backup, path)
            print(f"✅ Restored {path} from {latest_backup.name}")
        else:
            print(f"⚠️  No backup found for {path}")

    print()
    print("Rollback complete")


def main():
    parser = argparse.ArgumentParser(description="Upgrade evaluation to V2 (exploration-dependent)")
    parser.add_argument("--apply", action="store_true", help="Apply the upgrade")
    parser.add_argument("--rollback", action="store_true", help="Rollback to original")

    args = parser.parse_args()

    if args.apply:
        apply_upgrade()
    elif args.rollback:
        rollback_upgrade()
    else:
        parser.print_help()
        print()
        print("You must specify either --apply or --rollback")


if __name__ == "__main__":
    main()
