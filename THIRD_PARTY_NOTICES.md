# Third-party notices

This source-only package does not assign a project license or assert redistribution rights.

- CityFlow: `CityFlow/LICENSE.txt` (Apache 2.0); this is an inherited locally modified version,
  not a pristine upstream checkout. It includes startup/shutdown modifications.
- pybind11: `CityFlow/extern/pybind11/LICENSE`.
- RapidJSON: `CityFlow/extern/rapidjson/license.txt`.
- Milo dtoa helper: `CityFlow/extern/milo/LICENSE.txt`; inherited header has local changes.
- License retrieval provenance: `docs/third_party_file_provenance.json`.
- Teacher implementations under `src/agent/` are inherited from the research workspace.
  Original code attribution and applicable licenses require author confirmation.
- Roadnets and traffic inputs under `data/` are copied from the submission artifact.
  Original dataset attribution and redistribution terms require author confirmation.
- Python dependencies are installed separately and retain their own licenses.

No pretrained checkpoints, archived rollouts or generated control programs are shipped.
Confirm the outstanding project, teacher-code and dataset licensing before public distribution.
