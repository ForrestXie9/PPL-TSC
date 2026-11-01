#!/usr/bin/env python3
"""Check source-only packaging and all public method configuration templates, offline."""
import ast,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
def main():
 from ppl_tsc_current.profile import validate_config
 errors=[];files=[]
 for p in ROOT.rglob('*'):
  if not p.is_file():continue
  rel=p.relative_to(ROOT)
  if any(x in rel.parts for x in ['.venv','outputs','__pycache__']):continue
  files.append(p)
  if p.is_symlink():errors.append(str(rel)+': symlink')
  if p.suffix in ['.pth','.pt','.pkl','.pickle','.so'] or p.name in ['.env','api.local.json']:errors.append(str(rel)+': unexpected private/generated artifact')
  if p.suffix=='.py' and rel.parts[0]!='CityFlow':ast.parse(p.read_text(),filename=str(rel))
 for p in (ROOT/'configs/examples').glob('*.json'):validate_config(json.loads(p.read_text()))
 for city in (ROOT/'data').iterdir():
  if city.is_dir():
   for name in ['roadnet.json']+[f'flow_{i}.json' for i in range(5)]:
    json.loads((city/name).read_text())
 if errors:raise SystemExit('\n'.join(errors))
 print(json.dumps({'status':'passed','files_checked':len(files),'templates':len(list((ROOT/'configs/examples').glob('*.json'))),'API_calls':0}))
if __name__=='__main__':main()
