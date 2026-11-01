import importlib.util,json,os,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
spec=importlib.util.spec_from_file_location('offline_entry_runner',ROOT/'run.py');runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
from agent.imitation_based import llm_integration as api
class OfflineProviderTests(unittest.TestCase):
    def test_legacy_backend_cannot_bypass_offline_flag(self):
        for backend in ['comp','azure','azure_apim','comp_openai']:
            with patch.dict(os.environ,{'LLM_BACKEND':backend,'LLM_CACHE_MODE':'readonly','LLM_EXTRA_HEADERS':''}),patch.object(api.importlib,'import_module') as load:
                with self.assertRaisesRegex(ValueError,'refusing network'):
                    api.run_llm('fixture',api_endpoint='example.invalid',api_key='fake',model='fixture')
                load.assert_not_called()
    def test_json_provider_overrides_stale_backend_environment(self):
        with tempfile.TemporaryDirectory() as d,patch.dict(os.environ,{'LLM_BACKEND':'comp','METHOD_PROJECT_ROOT':str(ROOT)}):
            d=Path(d);trajectory=d/'source.jsonl'
            trajectory.write_text(json.dumps({'meta':{'model':'MPLight','dataset':'Hangzhou1','flow_idx':0,'direction_feature_semantics':'compatible_shared_lane_mean_v1'},'config':{'num_step':3600}})+'\n')
            cfg=json.loads((ROOT/'configs/examples/MPLight.json').read_text());cfg['data']['teacher_data']=str(trajectory);cfg['experiment']['output_dir']=str(d/'output');cfg['llm']['cache_dir']=str(d/'cache')
            runner._build_internal_argv(cfg,offline=True,disable_cache=False,validate_only=True)
            self.assertEqual(os.environ['LLM_BACKEND'],'openai_compatible');self.assertEqual(os.environ['LLM_CACHE_MODE'],'readonly')
if __name__=='__main__':unittest.main()
