# Final results

- `cargo test -p gen-zero-lod -p gen-zero-service`: exit **0**, **511 passed / 0 failed / 7 ignored**.
- Source SHA-256 comparison (138 Rust/config files): exit **0**.
- Focused graph and engine recovery tests: exit **0** each.
- Final debug warm-cache graph restore, 1024 nodes / 1 CSR edge / 1 pending edge: **20 samples**, min **33.582ms**, median **42.536ms**, max **48.896ms**; **0** samples >=50ms.
- Three-node engine restart: **1.567ms**, one sample.
- Local `cargo fmt --all -- --check` and `git diff --check`: exit **0**.

Final full test output tail:

```text
test contact_what_if_audit_and_latent_decide_run_on_the_contact_prior ... ok

test result: ok. 35 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.18s

   Doc-tests gen_zero_lod

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

   Doc-tests gen_zero_service

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

mbx[cache]: 0 hits, 0 misses, 21 incremental, 2 bypassed; 0 B downloaded, 0 B uploaded, 0 B stored locally
mbx[savings]: 1903 compilations served from cache to date
```
