"""Dated post-training handoff; retain original training-stage report bytes."""
import datetime
import hashlib
import json
import pathlib

R = pathlib.Path(__file__).resolve().parents[1]
ROOT = R.parent


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    receipt = R / 'PROGRESS_REPORT_MANIFEST_V2.json'
    assert not receipt.exists()
    fits = [read(p) for p in sorted((R / 'outputs/fits').glob('*/fit_summary.json'))]
    assert len(fits) == 8
    summary = read(R / 'outputs/eval_v1/summary.json')
    supervisor = read(R / 'outputs/evaluation_supervisor_v2.json')
    seal = read(R / 'outputs/COST_RULE_SEAL.json')
    opening = read(R / 'outputs/eval_v1/OPENING_RECEIPT.json')
    assert seal['utc'][:19] < opening['utc'][:19]
    assert seal['supplement_sha256'] == sha(R / 'MATCHED_COST_SUPPLEMENT.md')
    utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
    update = f'''# C49 post-training progress snapshot

UTC {utc}. All8 fits complete, {sum(f['updates'] for f in fits)} formal updates. Primary scoring has produced{summary['raw_groups']} raw groups /{summary['model_context_reveal_entries']} model-context-reveal entries, from512 independent synthetic physical paths and1536 paired density contexts.0 real domains. This is scored evidence pending independent verification, not a scientific improvement claim.

Authoritative resumed supervisor: outputs/evaluation_supervisor_v2.json, PID{supervisor['pid']}, state{supervisor['status']}, active{supervisor.get('active_script')}, child{supervisor.get('child_pid')}. Actual process handles are checked separately; a state file alone is not proof of a live job.

The original verifier repeated large compressed-array decoding per batch. Its precisely identified actual worker was terminated after a versioned I/O repair had been uploaded/hash-checked. Only raw/probe array caching changes in verify49_v2.py; all arithmetic, tolerances, trained weights and completed primary scores are retained. Original supervisor failures remain visible; they reflect this intentional verification replacement. See VERIFIER_IO_REPAIR.md and outputs/VERIFIER_IO_REPAIR_RECEIPT.json. No fitting or primary scoring was restarted.

The pre-EVAL-sealed60/120/180-second comparison rule is in MATCHED_COST_SUPPLEMENT.md. The resumed supervisor continues verify49_v2 ->cost49 ->analyze49 ->plot49. Analysis and figures must not be claimed complete before their actual receipts exist. New effect sizes remain withheld until independent verification passes.

One fit hit the maximum update budget;{sum(f['selected_at_final_update'] for f in fits)} selected their final update. Training/selection limits remain part of attribution. A fixed-budget frontier does not establish complete convergence or independent real-domain utility.

New closest priors include2026 convMMD/deconvolution generative learning and OASIS. The implemented score is an observation-kernel integration of an existing proper score. Distinct useful conditional state/dynamics learning, mechanism attribution, competent controls and fresh real-domain evidence are still required. See NOVELTY_CHALLENGE_20261017.md and LITERATURE_SUPPLEMENT_20261017.md.

The original seven progress reports and CURRENT_STATE.md remain dated training-stage snapshots; this file supersedes their live-count interpretation without modifying their preserved manifest. Accepted final research directions0. Overall goal ACTIVE_UNACHIEVED. C48 is complete/deprioritized and must not be restarted.
'''
    (R / 'CURRENT_STATE_V2.md').write_text(update, encoding='utf-8')
    archive = R / 'outputs/root_history_v2'
    archive.mkdir()
    root_records = []
    header = f'''<!-- cycle50-progress-index-start -->
# C49收尾：8组训练和主评分完成，独立复核与同预算比较继续

UTC {utc}。18800正式updates；主评分16组原始轨迹/12288模型—观测历史—揭示分数，对应512条独立合成物理轨迹、1536个配对观测历史。独立复核仍在执行，不作方法收益主张。验证器的大型压缩数组重复读取已按版本修复，原权重和主评分未重跑。

同预算60/120/180秒规则已在EVAL打开前留档。恢复队列自动继续独立NumPy复核、固定预算比较、分析与图。最新状态见[当前快照](cycle49/CURRENT_STATE_V2.md)及实际进程；旧训练快照作为历史保存。已有convMMD/观测感知生成推断使创新边界更紧，见[近邻挑战](cycle49/NOVELTY_CHALLENGE_20261017.md)。尚未建立新框架贡献，接受最终方向0，整体ACTIVE_UNACHIEVED。

以下为历史。
<!-- cycle50-progress-index-end -->

'''
    for name in ['CURRENT_RESEARCH_STATUS.md', 'literature_review.md', 'baseline_registry.md', 'experimental_results.md',
                 'negative_evidence.md', 'hypothesis_space.md', 'research_ideas.md', 'final_research_direction.md']:
        path = ROOT / name
        previous = path.read_bytes()
        (archive / name).write_bytes(previous)
        assert path.read_bytes() == previous
        path.write_bytes(header.encode('utf-8') + previous)
        root_records.append({'path': name, 'before': hashlib.sha256(previous).hexdigest(), 'after': sha(path)})
    value = {'utc': utc, 'status': 'POST_TRAINING_VERIFICATION_AND_COST_EXTENSION_PENDING', 'formal_fits': 8,
             'formal_updates': sum(f['updates'] for f in fits), 'primary_score_sha256': sha(R / 'outputs/eval_v1/summary.json'),
             'new_current_state_sha256': sha(R / 'CURRENT_STATE_V2.md'), 'root_index': root_records,
             'source_sha256': sha(pathlib.Path(__file__)), 'goal_complete': False}
    receipt.write_text(json.dumps(value, indent=2), encoding='utf-8')
    print(json.dumps({'status': value['status'], 'formal_updates': value['formal_updates'], 'goal_complete': False}), flush=True)


if __name__ == '__main__':
    main()
