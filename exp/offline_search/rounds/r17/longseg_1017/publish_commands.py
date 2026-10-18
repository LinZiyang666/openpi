"""Add fresh-name wrappers in the existing chain runner's command directory."""
from .audit import HERE

def main():
    target=HERE.parent/'filter_failed_knob/commands';count=0
    for source in (HERE/'commands').glob('*.sh'):
        assert source.name.startswith(('prepare_fa_ls_','launch_fa_ls_'))
        path=target/source.name
        body='#!/usr/bin/env bash\nset -euo pipefail\nexec bash '+str(source)+'\n'
        if path.exists():assert path.read_text()==body,'Never replace an existing different queue script'
        else:path.write_text(body);path.chmod(0o755)
        count+=1
    print(count,'fresh-name wrappers; existing command files unchanged')
if __name__=='__main__':main()
