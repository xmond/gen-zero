use gen_zero_worldmodel::{
    compress_contact_trajectory_zstd, compress_phase_trajectory_zstd,
    decompress_contact_trajectory_zstd, decompress_phase_trajectory_zstd, ContactState, PhaseState,
    WorldModelError,
};

fn contact_bits_equal<const N: usize>(a: &ContactState<N>, b: &ContactState<N>) {
    for (x, y) in
        a.q.iter()
            .chain(a.p.iter())
            .chain(std::iter::once(&a.s))
            .zip(b.q.iter().chain(b.p.iter()).chain(std::iter::once(&b.s)))
    {
        assert_eq!(x.to_bits(), y.to_bits());
    }
}
fn phase_bits_equal<const N: usize>(a: &PhaseState<N>, b: &PhaseState<N>) {
    for (x, y) in
        a.q.iter()
            .chain(a.p.iter())
            .zip(b.q.iter().chain(b.p.iter()))
    {
        assert_eq!(x.to_bits(), y.to_bits());
    }
}

#[test]
fn single_state_round_trip_preserves_bits() {
    let c = ContactState::<2>::new([-0.0, 1.25], [f32::MIN_POSITIVE, -3.5], -0.0);
    contact_bits_equal(
        &c,
        &ContactState::decompress_zstd(&c.compress_zstd(3).unwrap()).unwrap(),
    );
    let p = PhaseState::<2>::new([-0.0, 1.25], [f32::MIN_POSITIVE, -3.5]);
    phase_bits_equal(
        &p,
        &PhaseState::decompress_zstd(&p.compress_zstd(3).unwrap()).unwrap(),
    );
}

#[test]
fn hundred_step_trajectory_round_trip_and_compression() {
    let contacts: Vec<_> = (0..100)
        .map(|i| ContactState::<16>::new([i as f32; 16], [-0.0; 16], 1.0))
        .collect();
    let encoded = compress_contact_trajectory_zstd(&contacts, 3).unwrap();
    assert!(
        encoded.len() * 2 <= contacts.len() * 33 * 4,
        "{} bytes",
        encoded.len()
    );
    for (a, b) in contacts.iter().zip(
        decompress_contact_trajectory_zstd::<16>(&encoded)
            .unwrap()
            .iter(),
    ) {
        contact_bits_equal(a, b);
    }
    let phases: Vec<_> = (0..100)
        .map(|i| PhaseState::<16>::new([i as f32; 16], [-0.0; 16]))
        .collect();
    let encoded = compress_phase_trajectory_zstd(&phases, 3).unwrap();
    assert!(
        encoded.len() * 2 <= phases.len() * 32 * 4,
        "{} bytes",
        encoded.len()
    );
    for (a, b) in phases.iter().zip(
        decompress_phase_trajectory_zstd::<16>(&encoded)
            .unwrap()
            .iter(),
    ) {
        phase_bits_equal(a, b);
    }
}

#[test]
fn malformed_streams_fail_closed() {
    let c = ContactState::<2>::new([1.0, 2.0], [3.0, 4.0], 5.0);
    let encoded = c.compress_zstd(3).unwrap();
    for stream in [
        &b"garbage"[..],
        &encoded[..encoded.len() - 1],
        &encoded[..22],
    ] {
        assert!(matches!(
            ContactState::<2>::decompress_zstd(stream),
            Err(WorldModelError::Compression(_))
        ));
    }
    let mut corrupt = encoded.clone();
    corrupt[27] ^= 0x80;
    assert!(matches!(
        ContactState::<2>::decompress_zstd(&corrupt),
        Err(WorldModelError::Compression(_))
    ));
    assert!(matches!(
        ContactState::<3>::decompress_zstd(&encoded),
        Err(WorldModelError::Compression(_))
    ));
    assert!(matches!(
        PhaseState::<2>::decompress_zstd(&encoded),
        Err(WorldModelError::Compression(_))
    ));
    let p = PhaseState::<2>::new([1.0, 2.0], [3.0, 4.0]);
    let phase = p.compress_zstd(3).unwrap();
    assert!(matches!(
        PhaseState::<2>::decompress_zstd(&phase[..phase.len() - 1]),
        Err(WorldModelError::Compression(_))
    ));
}

#[test]
fn non_finite_values_fail_closed() {
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(matches!(
            ContactState::<1>::new([value], [0.0], 0.0).compress_zstd(3),
            Err(WorldModelError::Compression(_))
        ));
        assert!(matches!(
            ContactState::<1>::new([0.0], [value], 0.0).compress_zstd(3),
            Err(WorldModelError::Compression(_))
        ));
        assert!(matches!(
            ContactState::<1>::new([0.0], [0.0], value).compress_zstd(3),
            Err(WorldModelError::Compression(_))
        ));
        assert!(matches!(
            PhaseState::<1>::new([value], [0.0]).compress_zstd(3),
            Err(WorldModelError::Compression(_))
        ));
        assert!(matches!(
            PhaseState::<1>::new([0.0], [value]).compress_zstd(3),
            Err(WorldModelError::Compression(_))
        ));
    }
}

#[test]
fn header_length_and_appended_bytes_are_rejected() {
    let state = ContactState::<1>::new([1.0], [2.0], 3.0);
    let encoded = state.compress_zstd(3).unwrap();
    for count in [0_u64, 2, u64::MAX] {
        let mut changed = encoded.clone();
        changed[14..22].copy_from_slice(&count.to_le_bytes());
        assert!(matches!(
            ContactState::<1>::decompress_zstd(&changed),
            Err(WorldModelError::Compression(_))
        ));
    }
    let mut appended = encoded.clone();
    appended.extend_from_slice(b"junk");
    assert!(matches!(
        ContactState::<1>::decompress_zstd(&appended),
        Err(WorldModelError::Compression(_))
    ));
}

#[test]
fn empty_trajectory_and_zero_dimension_contract() {
    let encoded = compress_contact_trajectory_zstd::<2>(&[], 3).unwrap();
    assert!(decompress_contact_trajectory_zstd::<2>(&encoded)
        .unwrap()
        .is_empty());
    let encoded = compress_phase_trajectory_zstd::<2>(&[], 3).unwrap();
    assert!(decompress_phase_trajectory_zstd::<2>(&encoded)
        .unwrap()
        .is_empty());
    assert!(matches!(
        PhaseState::<0>::zeros().compress_zstd(3),
        Err(WorldModelError::Compression(_))
    ));
}
