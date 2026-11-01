"""A private /tmp must not hide a virtual environment located beneath it."""
import sys,unittest,tempfile
from pathlib import Path
from unittest.mock import patch,MagicMock
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from LLM_SR_DISTILLATION.policy_process import PolicyProcess
class RuntimeMountTests(unittest.TestCase):
 def test_tmpfs_precedes_readonly_venv_mount(self):
  with tempfile.TemporaryDirectory() as venv:
   child=MagicMock();child.poll.return_value=0
   with patch('sys.prefix',venv),patch('shutil.which',return_value='/usr/bin/bwrap'),patch('subprocess.Popen',return_value=child) as popen,patch.object(PolicyProcess,'_request',return_value={'value':0}),patch.dict('os.environ',{'PPL_POLICY_ISOLATION':'sandbox'}):
    worker=PolicyProcess('def compute_utility(state, phase_id=None, feature_stats=None):\n    return 0.0\n')
    try:
     command=popen.call_args.args[0]
     self.assertLess(command.index('--tmpfs'),command.index(venv))
     self.assertEqual(command[command.index(venv)-1],'--ro-bind')
     self.assertIn('--unshare-all',command)
    finally:worker.close()
if __name__=='__main__':unittest.main()
