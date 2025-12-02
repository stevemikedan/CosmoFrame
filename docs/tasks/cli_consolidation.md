# CosmoSim Unified CLI + Viewer Consolidation

## Phase 1: Canonicalize cosmosim.py
- [ ] Add missing CLI arguments: `--dt`, `--topology`, `--substrate`, `--expansion`, `--seed`, `--entities` (alias `-N`), `--view`.
- [ ] Implement scenario loading logic:
    - [ ] Auto-prefix "scenarios." if missing.
    - [ ] Dynamic import.
    - [ ] Fallback if no scenario provided.

## Phase 2: Viewer Routing Logic
- [ ] Implement `--view` routing:
    - [ ] `auto`: Default. Maps to debug if interactive, web if export, else none.
    - [ ] `debug`: Runs `viewer/viewer.py`.
    - [ ] `web`: Enforces export-json, prints instructions for `viewer/test.html`.
    - [ ] `none`: Silent run (headless).
- [ ] Add runtime console banner.

## Phase 3: Legacy Marking
- [ ] Add legacy warning to `run_sim.py`.
- [ ] Add legacy warning to `jit_run_sim.py`.

## Phase 4: Documentation Updates
- [ ] Update root `README.md` with "How to Run" section.
- [ ] Update `viewer/README.md` with viewer distinction.
- [ ] Create `docs/ARCHITECTURE_VIEWERS.md` with diagram and explanation.

## Phase 5: Validation
- [ ] Run `pytest -q`.
- [ ] Verify DEBUG mode (`--view debug`).
- [ ] Verify WEB mode (`--view web`).
- [ ] Verify HEADLESS mode (`--view none`).
