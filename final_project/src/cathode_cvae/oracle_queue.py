"""Read claim identities once per queue loop; never inspect or reclaim results."""
import os


def unclaimed_tasks(tasks, claim_root, model):
    """Reserve every existing name, including completed and failed claims.

    This is a non-atomic snapshot: concurrent workers may select the same
    candidate. The caller must retain its atomic mkdir and FileExistsError
    handling. It must never delete failed, existing, or abandoned claims.
    """
    claimed = set(os.listdir(claim_root))
    return [task for task in tasks
            if task['model'] == model and task['task_id'] not in claimed]
