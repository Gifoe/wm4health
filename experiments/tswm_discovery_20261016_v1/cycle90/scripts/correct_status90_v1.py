from common90_v1 import *

def main():
    old=read(O/'EVIDENCE_CLOSURE90_V1.json')
    phrase='No further CSDI/MLP tuning is authorized by a positive interpretation of these results.'
    replacement='These results do not justify further tuning of this CSDI/MLP configuration.'
    before=(ROOT/'CURRENT_RESEARCH_STATUS.md').read_bytes();assert sha(ROOT/'CURRENT_RESEARCH_STATUS.md')==old['current_canonical_sha256']
    with (ROOT/'CURRENT_RESEARCH_STATUS_V90_BEFORE_LANGUAGE_FIX.md').open('xb') as f:f.write(before)
    text=before.decode('utf-8');assert phrase in text;fixed=text.replace(phrase,replacement)
    for path in [ROOT/'CURRENT_RESEARCH_STATUS.md',ROOT/'CURRENT_RESEARCH_STATUS_V90.md']:
        assert path.read_bytes()==before;path.write_text(fixed,encoding='utf-8')
    revised=dict(old,utc=utc(),current_canonical_sha256=sha(ROOT/'CURRENT_RESEARCH_STATUS.md'),supersedes_receipt_sha256=sha(O/'EVIDENCE_CLOSURE90_V1.json'),language_correction='A scientific stop decision is not a revocation of user research authorization; no evidence or results changed')
    write(O/'EVIDENCE_CLOSURE90_V2.json',revised)
    print(json.dumps(dict(status='STATUS_LANGUAGE_CLARIFIED_RESULTS_UNCHANGED',canonical_sha256=revised['current_canonical_sha256'])),flush=True)

if __name__=='__main__':main()
