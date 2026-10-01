use super::*;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;

fn tag(n: u8) -> Digest {
    sha256(&[b"test-digest", &[n]])
}

fn digests(seed: u8) -> AssetDigests {
    AssetDigests {
        model: tag(seed),
        geometry: tag(seed + 1),
        atlas: tag(seed + 2),
        graph: tag(seed + 3),
        policy: tag(seed + 4),
    }
}

fn key() -> MountKey {
    MountKey::new("acme", "main")
}

fn genesis(k: MountKey) -> MountSnapshot {
    MountSnapshot::genesis(k, digests(10), 0, Arc::from(&b"assets-v1"[..])).unwrap()
}

fn setup() -> (AtomicMountRegistry, Arc<Snapshot>) {
    let registry = AtomicMountRegistry::new();
    let base = registry.register(genesis(key())).unwrap();
    (registry, base)
}

fn budget() -> Budget {
    Budget {
        max_steps: 16,
        max_time_ns: u64::MAX,
        max_bytes: 1 << 20,
        residual_limit: 0.5,
        numeric_error: 1e-9,
        policy: tag(200),
    }
}

fn atlas_change(seed: u8) -> SnapshotChange {
    SnapshotChange {
        atlas: Some(tag(seed)),
        ..SnapshotChange::default()
    }
}

/// Proposal plus a candidate derived from `base` with `change`.
fn propose(base: &Arc<Snapshot>, change: &SnapshotChange) -> (Proposal, CandidateMount) {
    let proposal = Proposal::new(Arc::clone(base), Arc::from(&b"refine"[..])).unwrap();
    let next = Arc::new(base.derive(change).unwrap());
    let candidate = CandidateMount::new(next, &proposal);
    (proposal, candidate)
}

fn sealed(
    registry: &AtomicMountRegistry,
    base: &Arc<Snapshot>,
    seed: u8,
) -> Result<ValidatedMount> {
    let (proposal, candidate) = propose(base, &atlas_change(seed));
    registry.validate(proposal, candidate, &budget())
}

/// One full slow-loop publication from whatever is mounted now.
fn publish_atlas(registry: &AtomicMountRegistry, seed: u8) -> Result<Published> {
    let base = registry.load(&key())?;
    let sealed = sealed(registry, &base, seed)?;
    registry.compare_and_mount(&key(), sealed)
}

// ---- digest ----

#[test]
fn digest_is_real_sha256() {
    // FIPS 180-2 test vector for "abc".
    assert_eq!(
        digest_hex(&sha256(&[b"abc"])),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    assert_eq!(sha256(&[b"a", b"bc"]), sha256(&[b"abc"]));
}

#[test]
fn snapshot_digest_covers_every_field() {
    let base = genesis(key());
    assert!(base.verify_digest());
    let mut seen = vec![*base.digest()];
    let changes = [
        SnapshotChange {
            model: Some(tag(1)),
            ..Default::default()
        },
        SnapshotChange {
            geometry: Some(tag(1)),
            ..Default::default()
        },
        SnapshotChange {
            atlas: Some(tag(1)),
            ..Default::default()
        },
        SnapshotChange {
            graph: Some(tag(1)),
            ..Default::default()
        },
        SnapshotChange {
            policy: Some(tag(1)),
            ..Default::default()
        },
        SnapshotChange {
            assets: Some(Arc::from(&b"assets-v2"[..])),
            ..Default::default()
        },
        SnapshotChange {
            watermark: Some(7),
            ..Default::default()
        },
    ];
    for change in &changes {
        let next = base.derive(change).unwrap();
        assert!(next.verify_digest());
        seen.push(*next.digest());
    }
    // A different key alone changes the digest too.
    seen.push(*genesis(MountKey::new("acme", "other")).digest());
    seen.push(*genesis(MountKey::new("acm", "emain")).digest());
    let unique: std::collections::HashSet<_> = seen.iter().collect();
    assert_eq!(
        unique.len(),
        seen.len(),
        "two different contents share a digest"
    );
}

#[test]
fn a_tampered_snapshot_fails_digest_verification() {
    let mut snap = genesis(key());
    snap.watermark += 1;
    assert!(!snap.verify_digest());
}

#[test]
fn genesis_refuses_zero_digests_and_empty_keys() {
    let mut zeroed = digests(10);
    zeroed.atlas = [0; 32];
    let bad = MountSnapshot::genesis(key(), zeroed, 0, Arc::from(Vec::new()));
    assert_eq!(bad.unwrap_err(), Reject::InvalidCertificate);
    let empty = MountSnapshot::genesis(
        MountKey::new("", "w"),
        digests(10),
        0,
        Arc::from(Vec::new()),
    );
    assert_eq!(empty.unwrap_err(), Reject::InvalidCertificate);
}

// ---- monotonic versions and immutability ----

#[test]
fn version_rises_by_one_per_mount_and_old_snapshots_stay_intact() {
    let (registry, v1) = setup();
    assert_eq!(v1.version(), Version(1));
    let (old_digest, old_atlas) = (*v1.digest(), v1.epochs().atlas);

    let mut previous = Version(1);
    for seed in 50..55u8 {
        let published = publish_atlas(&registry, seed).unwrap();
        assert_eq!(published.from, previous);
        assert_eq!(published.to, Version(previous.0 + 1));
        let live = registry.load(&key()).unwrap();
        assert_eq!(live.version(), published.to);
        assert_eq!(*live.digest(), published.digest);
        assert!(live.verify_digest());
        previous = published.to;
    }
    assert_eq!(previous, Version(6));

    // The reader that captured v1 still holds v1, byte for byte.
    assert_eq!(v1.version(), Version(1));
    assert_eq!(*v1.digest(), old_digest);
    assert_eq!(v1.epochs().atlas, old_atlas);
    assert_eq!(v1.assets(), b"assets-v1");
    assert!(v1.verify_digest());
}

#[test]
fn every_asset_family_change_bumps_the_version() {
    let base = genesis(key());
    let changes = [
        SnapshotChange {
            model: Some(tag(1)),
            ..Default::default()
        },
        SnapshotChange {
            geometry: Some(tag(1)),
            ..Default::default()
        },
        SnapshotChange {
            atlas: Some(tag(1)),
            ..Default::default()
        },
        SnapshotChange {
            graph: Some(tag(1)),
            ..Default::default()
        },
        SnapshotChange {
            policy: Some(tag(1)),
            ..Default::default()
        },
        SnapshotChange {
            assets: Some(Arc::from(&b"x"[..])),
            ..Default::default()
        },
    ];
    for change in &changes {
        assert_eq!(base.derive(change).unwrap().version(), Version(2));
    }
}

#[test]
fn a_change_that_changes_nothing_is_stalled_and_the_watermark_never_regresses() {
    let base = genesis(key());
    assert_eq!(
        base.derive(&SnapshotChange::default()).unwrap_err(),
        Reject::Stalled
    );
    let same_atlas = SnapshotChange {
        atlas: Some(base.epochs().atlas),
        ..Default::default()
    };
    assert_eq!(base.derive(&same_atlas).unwrap_err(), Reject::Stalled);

    let ahead = base
        .derive(&SnapshotChange {
            watermark: Some(5),
            ..Default::default()
        })
        .unwrap();
    let back = SnapshotChange {
        watermark: Some(4),
        atlas: Some(tag(1)),
        ..Default::default()
    };
    assert_eq!(ahead.derive(&back).unwrap_err(), Reject::InvalidCertificate);
}

#[test]
fn version_overflow_is_refused_not_wrapped() {
    assert_eq!(
        Version(u64::MAX).next().unwrap_err(),
        Reject::BudgetExceeded
    );
    let epochs = Epochs {
        version: Version(u64::MAX),
        model: tag(1),
        geometry: tag(2),
        atlas: tag(3),
        graph: tag(4),
        policy: tag(5),
    };
    let top = MountSnapshot::seal(key(), epochs, 0, Arc::from(Vec::new())).unwrap();
    assert_eq!(
        top.derive(&atlas_change(9)).unwrap_err(),
        Reject::BudgetExceeded
    );
}

// ---- CAS ----

#[test]
fn cas_has_exactly_one_winner_among_racing_seals() {
    const RACERS: usize = 8;
    let (registry, base) = setup();
    let registry = Arc::new(registry);
    // Every racer seals a different candidate off the same base.
    let seals: Vec<ValidatedMount> = (0..RACERS)
        .map(|i| sealed(&registry, &base, 100 + i as u8).unwrap())
        .collect();
    let barrier = Arc::new(Barrier::new(RACERS));
    let handles: Vec<_> = seals
        .into_iter()
        .map(|seal| {
            let (registry, barrier) = (Arc::clone(&registry), Arc::clone(&barrier));
            thread::spawn(move || {
                barrier.wait();
                registry.compare_and_mount(&key(), seal)
            })
        })
        .collect();
    let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();

    let winners: Vec<_> = results.iter().filter_map(|r| r.as_ref().ok()).collect();
    assert_eq!(winners.len(), 1, "results: {results:?}");
    let losers = results
        .iter()
        .filter(|r| **r == Err(Reject::CasConflict))
        .count();
    assert_eq!(losers, RACERS - 1, "results: {results:?}");

    let live = registry.load(&key()).unwrap();
    assert_eq!(live.version(), Version(2));
    assert_eq!(*live.digest(), winners[0].digest);

    // A loser cannot reuse its seal: a fresh validate on the stale base fails.
    let stale = sealed(&registry, &base, 100);
    assert_eq!(stale.unwrap_err(), Reject::CasConflict);
    assert_eq!(registry.load(&key()).unwrap().version(), Version(2));
}

#[test]
fn concurrent_writers_and_readers_keep_one_linear_history() {
    const WRITERS: usize = 4;
    const PER_WRITER: usize = 20;
    let (registry, _) = setup();
    let registry = Arc::new(registry);
    let done = Arc::new(AtomicBool::new(false));
    let conflicts = Arc::new(AtomicUsize::new(0));

    let readers: Vec<_> = (0..3)
        .map(|_| {
            let (registry, done) = (Arc::clone(&registry), Arc::clone(&done));
            thread::spawn(move || {
                let mut last = Version(0);
                while !done.load(Ordering::Acquire) {
                    let snap = registry.load(&key()).unwrap();
                    assert!(snap.version() >= last, "a reader saw the version go back");
                    assert!(snap.verify_digest(), "a reader saw a torn snapshot");
                    last = snap.version();
                }
            })
        })
        .collect();

    let writers: Vec<_> = (0..WRITERS)
        .map(|w| {
            let (registry, conflicts) = (Arc::clone(&registry), Arc::clone(&conflicts));
            thread::spawn(move || {
                let mut published = Vec::new();
                for i in 0..PER_WRITER {
                    let seed = 100 + (w * PER_WRITER + i) as u8;
                    loop {
                        match publish_atlas(&registry, seed) {
                            Ok(p) => {
                                published.push(p);
                                break;
                            }
                            Err(Reject::CasConflict) => {
                                conflicts.fetch_add(1, Ordering::Relaxed);
                            }
                            Err(other) => panic!("unexpected reject: {other}"),
                        }
                    }
                }
                published
            })
        })
        .collect();

    let mut all: Vec<Published> = writers
        .into_iter()
        .flat_map(|h| h.join().unwrap())
        .collect();
    done.store(true, Ordering::Release);
    for r in readers {
        r.join().unwrap();
    }

    let total = WRITERS * PER_WRITER;
    assert_eq!(all.len(), total);
    // No two publications share a base and there is no gap: one linear chain.
    all.sort_by_key(|p| p.from);
    for (i, p) in all.iter().enumerate() {
        assert_eq!(p.from, Version(1 + i as u64));
        assert_eq!(p.to, Version(2 + i as u64));
    }
    let live = registry.load(&key()).unwrap();
    assert_eq!(live.version(), Version(1 + total as u64));
    assert_eq!(live.digest(), &all.last().unwrap().digest);
    eprintln!(
        "cas conflicts retried during the stress run: {}",
        conflicts.load(Ordering::Relaxed)
    );
}

#[test]
fn concurrent_registration_of_one_key_has_one_winner() {
    let registry = Arc::new(AtomicMountRegistry::new());
    let barrier = Arc::new(Barrier::new(6));
    let handles: Vec<_> = (0..6)
        .map(|_| {
            let (registry, barrier) = (Arc::clone(&registry), Arc::clone(&barrier));
            thread::spawn(move || {
                barrier.wait();
                registry.register(genesis(key())).map(|_| ())
            })
        })
        .collect();
    let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    assert_eq!(results.iter().filter(|r| r.is_ok()).count(), 1);
    assert!(results
        .iter()
        .filter(|r| r.is_err())
        .all(|r| *r == Err(Reject::CasConflict)));
}

#[test]
fn unknown_keys_and_mismatched_seals_are_refused() {
    let (registry, base) = setup();
    let missing = MountKey::new("nobody", "nowhere");
    assert_eq!(registry.load(&missing).unwrap_err(), Reject::CoverageLost);

    let other = registry
        .register(genesis(MountKey::new("acme", "second")))
        .unwrap();
    assert_eq!(other.version(), Version(1));
    let seal = sealed(&registry, &base, 60).unwrap();
    // A seal for `acme/main` cannot be mounted under another key.
    let wrong = registry.compare_and_mount(other.key(), seal);
    assert_eq!(wrong.unwrap_err(), Reject::EpochMismatch);
    assert_eq!(registry.load(&key()).unwrap().version(), Version(1));
    assert_eq!(registry.load(other.key()).unwrap().version(), Version(1));
}

#[test]
fn a_request_stays_on_one_generation_and_foreign_epochs_are_refused() {
    let (registry, _) = setup();
    let binding = RequestBinding::capture(&registry, &key()).unwrap();
    let v1_epochs = binding.snapshot().epochs().clone();
    binding.admit(&v1_epochs).unwrap();
    binding.ensure_current(&registry).unwrap();

    publish_atlas(&registry, 80).unwrap();
    let v2 = registry.load(&key()).unwrap();

    // The in-flight request keeps reading generation 1.
    assert_eq!(binding.snapshot().version(), Version(1));
    assert!(binding.snapshot().verify_digest());
    binding.admit(&v1_epochs).unwrap();
    // An artefact from generation 2 does not mix in.
    assert_eq!(
        binding.admit(v2.epochs()).unwrap_err(),
        Reject::EpochMismatch
    );
    // A commit point that needs the live generation sees the request is stale.
    assert_eq!(
        binding.ensure_current(&registry).unwrap_err(),
        Reject::EpochMismatch
    );

    // Same version number but a different atlas digest is still a mismatch.
    let mut forged = v1_epochs.clone();
    forged.atlas = tag(99);
    assert_eq!(binding.admit(&forged).unwrap_err(), Reject::EpochMismatch);

    // A new request captures generation 2.
    let fresh = RequestBinding::capture(&registry, &key()).unwrap();
    assert_eq!(fresh.snapshot().version(), Version(2));
    fresh.ensure_current(&registry).unwrap();
}

#[test]
fn a_candidate_from_another_generation_or_key_is_refused() {
    let (registry, v1) = setup();
    publish_atlas(&registry, 82).unwrap();
    let v2 = registry.load(&key()).unwrap();

    // Proposal seen on v1, candidate built off v2 (so version 3).
    let proposal = Proposal::new(Arc::clone(&v1), Arc::from(&b"p"[..])).unwrap();
    let skewed = CandidateMount::new(Arc::new(v2.derive(&atlas_change(83)).unwrap()), &proposal);
    assert_eq!(
        registry.validate(proposal, skewed, &budget()).unwrap_err(),
        Reject::EpochMismatch
    );

    // A version jump of two is refused even off the right base.
    let proposal = Proposal::new(Arc::clone(&v2), Arc::from(&b"p"[..])).unwrap();
    let jumped_epochs = Epochs {
        version: Version(4),
        ..v2.epochs().clone()
    };
    let jumped = MountSnapshot::seal(key(), jumped_epochs, 0, Arc::from(&b"j"[..])).unwrap();
    let candidate = CandidateMount::new(Arc::new(jumped), &proposal);
    assert_eq!(
        registry
            .validate(proposal, candidate, &budget())
            .unwrap_err(),
        Reject::EpochMismatch
    );

    // A candidate for another key is refused.
    let proposal = Proposal::new(Arc::clone(&v2), Arc::from(&b"p"[..])).unwrap();
    let foreign = genesis(MountKey::new("acme", "other"));
    let foreign_next = foreign.derive(&atlas_change(84)).unwrap();
    let candidate = CandidateMount::new(Arc::new(foreign_next), &proposal);
    assert_eq!(
        registry
            .validate(proposal, candidate, &budget())
            .unwrap_err(),
        Reject::EpochMismatch
    );
    assert_eq!(registry.load(&key()).unwrap().version(), Version(2));
}

// ---- validate really validates ----

#[test]
fn validate_rejects_bad_certificates_budgets_and_forgeries() {
    let (registry, base) = setup();
    let (proposal, candidate) = propose(&base, &atlas_change(90));

    // Candidate built for a different proposal.
    let other = Proposal::new(Arc::clone(&base), Arc::from(&b"other"[..])).unwrap();
    let wrong = CandidateMount::new(Arc::clone(candidate.next()), &other);
    assert_eq!(
        registry
            .validate(proposal.clone(), wrong, &budget())
            .unwrap_err(),
        Reject::InvalidCertificate
    );

    // Forged proposal digest.
    let mut forged = proposal.clone();
    forged.digest[0] ^= 1;
    let cand = CandidateMount {
        next: Arc::clone(candidate.next()),
        proposal: forged.digest,
    };
    assert_eq!(
        registry.validate(forged, cand, &budget()).unwrap_err(),
        Reject::InvalidCertificate
    );

    // Snapshot whose stored digest no longer matches its contents.
    let mut tampered = base.derive(&atlas_change(90)).unwrap();
    tampered.assets = Arc::from(&b"swapped"[..]);
    let cand = CandidateMount::new(Arc::new(tampered), &proposal);
    assert_eq!(
        registry
            .validate(proposal.clone(), cand, &budget())
            .unwrap_err(),
        Reject::InvalidCertificate
    );

    // Candidate that claims data nobody deposited.
    let claim = SnapshotChange {
        atlas: Some(tag(91)),
        watermark: Some(9),
        ..Default::default()
    };
    let (p, c) = propose(&base, &claim);
    assert_eq!(
        registry.validate(p, c, &budget()).unwrap_err(),
        Reject::InvalidCertificate
    );

    // Budget problems.
    let big = SnapshotChange {
        atlas: Some(tag(92)),
        assets: Some(vec![7u8; 4096].into()),
        ..Default::default()
    };
    let (p, c) = propose(&base, &big);
    let tight = Budget {
        max_bytes: 4095,
        ..budget()
    };
    assert_eq!(
        registry.validate(p, c, &tight).unwrap_err(),
        Reject::BudgetExceeded
    );
    for bad in [f64::NAN, f64::INFINITY, -1.0] {
        let (p, c) = propose(&base, &atlas_change(93));
        let b = Budget {
            residual_limit: bad,
            ..budget()
        };
        assert_eq!(
            registry.validate(p, c, &b).unwrap_err(),
            Reject::NonFiniteState
        );
    }
    let (p, c) = propose(&base, &atlas_change(93));
    let b = Budget {
        policy: [0; 32],
        ..budget()
    };
    assert_eq!(
        registry.validate(p, c, &b).unwrap_err(),
        Reject::InvalidCertificate
    );

    // None of the refusals changed the mount.
    assert_eq!(registry.load(&key()).unwrap().version(), Version(1));
    // The honest pair still validates and the seal binds base, next and policy.
    let seal = registry.validate(proposal, candidate, &budget()).unwrap();
    assert_eq!(seal.base_version(), Version(1));
    let other_policy = Budget {
        policy: tag(201),
        ..budget()
    };
    let (p, c) = propose(&base, &atlas_change(90));
    let other_seal = registry.validate(p, c, &other_policy).unwrap();
    assert_ne!(seal.validation_digest(), other_seal.validation_digest());
}

#[test]
fn validate_enforces_the_time_budget() {
    let (registry, base) = setup();
    // Hashing two 32 MiB snapshots cannot fit in one nanosecond.
    let change = SnapshotChange {
        atlas: Some(tag(95)),
        assets: Some(vec![1u8; 32 << 20].into()),
        ..Default::default()
    };
    let (p, c) = propose(&base, &change);
    let b = Budget {
        max_bytes: 64 << 20,
        max_time_ns: 1,
        ..budget()
    };
    assert_eq!(
        registry.validate(p, c, &b).unwrap_err(),
        Reject::BudgetExceeded
    );
}

// ---- tension and chart refinement ----

// ---- production seam: the engine binds and stamps ----

mod engine {
    use super::*;
    use crate::zero::PolymorphicZeroEngine;
    use serde_json::{json, Value};

    fn engine() -> PolymorphicZeroEngine {
        PolymorphicZeroEngine::new().with_semantic(None)
    }

    fn compact() -> Value {
        json!({"action": "compact", "text": "hello mount"})
    }

    #[tokio::test]
    async fn every_request_is_stamped_with_the_generation_it_ran_on() {
        let engine = engine();
        let key = MountKey::new(crate::zero::DEFAULT_TENANT, crate::zero::DEFAULT_WORKSPACE);
        let v1 = engine.mounts().load(&key).unwrap();

        let out = engine.execute(&compact()).await.unwrap();
        let mount = &out.meta["mount"];
        assert_eq!(mount["version"], 1);
        assert_eq!(mount["tenant"], "default");
        assert_eq!(mount["digest"], digest_hex(v1.digest()));
        assert_eq!(mount["digest"].as_str().unwrap().len(), 64);

        // Publish through the same registry the engine reads.
        let base = engine.mounts().load(&key).unwrap();
        let (p, c) = propose(&base, &atlas_change(120));
        let seal = engine.mounts().validate(p, c, &budget()).unwrap();
        engine.mounts().compare_and_mount(&key, seal).unwrap();

        let out = engine.execute(&compact()).await.unwrap();
        assert_eq!(out.meta["mount"]["version"], 2);
        assert_ne!(out.meta["mount"]["digest"], digest_hex(v1.digest()));
    }

    #[tokio::test]
    async fn a_tenant_without_a_mount_is_refused_never_served_from_the_default() {
        let engine = engine();
        let mut req = compact();
        req["tenant"] = json!("ghost");
        let out = engine.execute(&req).await.unwrap();
        assert!(out.is_error);
        let r = out.rejection.expect("typed refusal");
        assert_eq!((r.code.as_str(), r.http_status), ("CoverageLost", 404));

        for bad in [json!({"workspace": 7}), json!({"tenant": "  "})] {
            let mut req = compact();
            for (k, v) in bad.as_object().unwrap() {
                req[k] = v.clone();
            }
            let out = engine.execute(&req).await.unwrap();
            assert_eq!(out.rejection.unwrap().code, "InvalidParams");
        }
    }

    #[tokio::test]
    async fn a_named_mount_is_served_on_its_own_generation() {
        let registry = Arc::new(AtomicMountRegistry::new());
        registry.register(genesis(key())).unwrap();
        let engine = engine().with_mounts(Arc::clone(&registry));
        // The default key is not mounted in this registry.
        let out = engine.execute(&compact()).await.unwrap();
        assert_eq!(out.rejection.unwrap().code, "CoverageLost");
        let mut req = compact();
        req["tenant"] = json!("acme");
        req["workspace"] = json!("main");
        let out = engine.execute(&req).await.unwrap();
        assert_eq!(out.meta["mount"]["tenant"], "acme");
        assert_eq!(out.meta["mount"]["version"], 1);
    }

    #[tokio::test]
    async fn the_http_gateway_carries_the_mount_stamp() {
        use crate::server::McpServer;
        use axum::body::Body;
        use axum::http::Request;
        use tower::util::ServiceExt;

        let router = McpServer::build_router(Arc::new(engine()), None);
        let request = Request::post("/v1/decisions")
            .header("content-type", "application/json")
            .body(Body::from(compact().to_string()))
            .unwrap();
        let response = router.oneshot(request).await.unwrap();
        let bytes = axum::body::to_bytes(response.into_body(), 1 << 20)
            .await
            .unwrap();
        let body: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(body["mount"]["version"], 1, "{body}");
    }
}
