from concurrent.futures import ThreadPoolExecutor
import json

from cathode_cvae.oracle_queue import unclaimed_tasks


def test_snapshot_candidates_match_original_and_keep_failed_claims(tmp_path):
    tasks=[dict(task_id=name,model=model) for name,model in
           [('complete','uma'),('failed','uma'),('running','uma'),
            ('pending','uma'),('other','mace')]]
    for name in ['complete','failed','running']:
        (tmp_path/name).mkdir()
    (tmp_path/'failed'/'failure.json').write_text(json.dumps({'error':'preserved'}))
    (tmp_path/'complete'/'score.json').write_text('{}')
    old=[t for t in tasks if t['model']=='uma' and not (tmp_path/t['task_id']).exists()]
    assert unclaimed_tasks(tasks,tmp_path,'uma')==old==[tasks[3]]
    assert unclaimed_tasks(tasks,tmp_path,'mace')==[tasks[4]]
    assert (tmp_path/'failed'/'failure.json').is_file()


def test_atomic_mkdir_prevents_duplicate_claims_under_concurrency(tmp_path):
    tasks=[dict(task_id=f'task_{i:03}',model='uma') for i in range(64)]
    def worker(_):
        won=[]
        while True:
            pending=unclaimed_tasks(tasks,tmp_path,'uma')
            if not pending:return won
            candidate=pending[0]['task_id']
            try:(tmp_path/candidate).mkdir()
            except FileExistsError:continue
            won.append(candidate)
    with ThreadPoolExecutor(max_workers=16) as pool:
        winners=sum(pool.map(worker,range(16)),[])
    assert len(winners)==len(set(winners))==64
    assert unclaimed_tasks(tasks,tmp_path,'uma')==[]
