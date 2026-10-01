//! Nonce Hunter: multi-threaded early-reject nonce hunter for the Fiat-Shamir seed.
//! Evaluates candidate nonces across 9,024 test shots, aborting on the very first
//! batch (64 shots) with any classical mismatch, phase garbage, or ancilla leak.

use crate::circuit::{analyze_ops, Op, OperationType, QubitId, QubitOrBit, NO_BIT, NO_QUBIT, NO_REG};
use crate::sim::Simulator;
use crate::weierstrass_elliptic_curve::WeierstrassEllipticCurve;
use alloy_primitives::U256;
use sha3::{
    digest::{ExtendableOutput, Update, XofReader},
    Shake256,
};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

const SECP256K1_P: U256 = U256::from_limbs([
    0xFFFF_FFFE_FFFF_FC2F,
    0xFFFF_FFFF_FFFF_FFFF,
    0xFFFF_FFFF_FFFF_FFFF,
    0xFFFF_FFFF_FFFF_FFFF,
]);

const SECP256K1_GX: U256 = U256::from_limbs([
    0x59F2_815B_16F8_1798,
    0x029B_FCDB_2DCE_28D9,
    0x55A0_6295_CE87_0B07,
    0x79BE_667E_F9DC_BBAC,
]);

const SECP256K1_GY: U256 = U256::from_limbs([
    0x9C47_D08F_FB10_D4B8,
    0xFD17_B448_A685_5419,
    0xD54F_BFC0_E110_8A8F,
    0x483A_DA77_26A3_C465,
]);

#[inline(always)]
fn sub_mod(a: U256, b: U256, m: U256) -> U256 {
    let a_m = a % m;
    let b_m = b % m;
    if a_m >= b_m {
        a_m - b_m
    } else {
        m - (b_m - a_m)
    }
}

#[derive(Clone, Copy, Debug)]
struct JacobianPoint {
    x: U256,
    y: U256,
    z: U256,
}

impl JacobianPoint {
    #[inline(always)]
    fn infinity() -> Self {
        Self {
            x: U256::ZERO,
            y: U256::ZERO,
            z: U256::ZERO,
        }
    }

    #[inline(always)]
    fn is_infinity(&self) -> bool {
        self.z.is_zero()
    }

    #[inline(always)]
    fn from_affine(x: U256, y: U256) -> Self {
        if x.is_zero() && y.is_zero() {
            Self::infinity()
        } else {
            Self {
                x,
                y,
                z: U256::from(1),
            }
        }
    }

    fn to_affine(&self, m: U256) -> (U256, U256) {
        if self.is_infinity() {
            return (U256::ZERO, U256::ZERO);
        }
        let z_inv = self.z.inv_mod(m).expect("z_inv");
        let z_inv2 = z_inv.mul_mod(z_inv, m);
        let z_inv3 = z_inv2.mul_mod(z_inv, m);
        let x = self.x.mul_mod(z_inv2, m);
        let y = self.y.mul_mod(z_inv3, m);
        (x, y)
    }

    fn double(&self, m: U256) -> Self {
        if self.is_infinity() || self.y.is_zero() {
            return Self::infinity();
        }
        // a = 0 for secp256k1
        let a = self.x.mul_mod(self.x, m); // X^2
        let b = self.y.mul_mod(self.y, m); // Y^2
        let c = b.mul_mod(b, m); // B^2 = Y^4
        let s = self.x.mul_mod(b, m).mul_mod(U256::from(4), m); // 4 * X * Y^2
        let e = a.mul_mod(U256::from(3), m); // 3 * X^2
        let f = e.mul_mod(e, m); // E^2
        let two_s = s.mul_mod(U256::from(2), m);
        let nx = sub_mod(f, two_s, m); // F - 2S
        let eight_c = c.mul_mod(U256::from(8), m);
        let ny = sub_mod(e.mul_mod(sub_mod(s, nx, m), m), eight_c, m);
        let nz = self.y.mul_mod(self.z, m).mul_mod(U256::from(2), m);
        Self { x: nx, y: ny, z: nz }
    }


    fn add_jacobian(&self, other: &Self, m: U256) -> Self {
        if self.is_infinity() {
            return *other;
        }
        if other.is_infinity() {
            return *self;
        }
        let z1_2 = self.z.mul_mod(self.z, m);
        let z2_2 = other.z.mul_mod(other.z, m);
        let u1 = self.x.mul_mod(z2_2, m);
        let u2 = other.x.mul_mod(z1_2, m);
        let s1 = self.y.mul_mod(z2_2, m).mul_mod(other.z, m);
        let s2 = other.y.mul_mod(z1_2, m).mul_mod(self.z, m);
        if u1 == u2 {
            if s1 == s2 {
                return self.double(m);
            } else {
                return Self::infinity();
            }
        }
        let h = sub_mod(u2, u1, m);
        let i = h.mul_mod(h, m).mul_mod(U256::from(4), m);
        let j = h.mul_mod(i, m);
        let r = sub_mod(s2, s1, m).mul_mod(U256::from(2), m);
        let v = u1.mul_mod(i, m);
        let r2 = r.mul_mod(r, m);
        let two_v = v.mul_mod(U256::from(2), m);
        let nx = sub_mod(sub_mod(r2, j, m), two_v, m);
        let two_s1_j = s1.mul_mod(j, m).mul_mod(U256::from(2), m);
        let ny = sub_mod(r.mul_mod(sub_mod(v, nx, m), m), two_s1_j, m);
        let z1_plus_z2 = self.z.add_mod(other.z, m);
        let nz = sub_mod(sub_mod(z1_plus_z2.mul_mod(z1_plus_z2, m), z1_2, m), z2_2, m).mul_mod(h, m);
        Self { x: nx, y: ny, z: nz }
    }
}

fn fast_mul_g(k: U256) -> (U256, U256) {
    let m = SECP256K1_P;
    let mut res = JacobianPoint::infinity();
    let mut base = JacobianPoint::from_affine(SECP256K1_GX, SECP256K1_GY);
    let mut exp = k;
    while !exp.is_zero() {
        if exp.bit(0) {
            res = res.add_jacobian(&base, m);
        }
        base = base.double(m);
        exp >>= 1;
    }
    res.to_affine(m)
}

fn secp256k1() -> WeierstrassEllipticCurve {
    WeierstrassEllipticCurve {
        modulus: SECP256K1_P,
        a: U256::from(0),
        b: U256::from(7),
        gx: SECP256K1_GX,
        gy: SECP256K1_GY,
        order: U256::from_str_radix(
            "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141",
            16,
        )
        .unwrap(),
    }
}

pub fn hunt(ops: &[Op]) {
    let (total_qubits, num_bits, _, regs) = analyze_ops(ops.iter());
    assert_eq!(regs.len(), 4, "expected 4 registers, got {}", regs.len());
    for r in &regs {
        assert_eq!(r.len(), 256);
    }
    assert!(ops.len() >= 96, "op stream too short for nonce tail");
    let base_ops = &ops[..ops.len() - 96];

    eprintln!(
        "=== NONCE HUNTER STARTING: total_qubits={} num_bits={} base_ops={} ===",
        total_qubits,
        num_bits,
        base_ops.len()
    );

    // Precompute Shake256 up to base_ops:
    let t0 = Instant::now();
    let mut base_hasher = Shake256::default();
    base_hasher.update(b"quantum_ecc-fiat-shamir-v2");
    base_hasher.update(&(ops.len() as u64).to_le_bytes());
    for op in base_ops {
        base_hasher.update(&[op.kind as u8]);
        base_hasher.update(&op.q_control2.0.to_le_bytes());
        base_hasher.update(&op.q_control1.0.to_le_bytes());
        base_hasher.update(&op.q_target.0.to_le_bytes());
        base_hasher.update(&op.c_target.0.to_le_bytes());
        base_hasher.update(&op.c_condition.0.to_le_bytes());
        base_hasher.update(&op.r_target.0.to_le_bytes());
    }
    eprintln!("Precomputed base hasher in {:.2?}", t0.elapsed());

    let start_nonce: u64 = std::env::var("NONCE_START")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);

    let max_scan: u64 = std::env::var("NONCE_COUNT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5_000_000);

    let num_threads: usize = std::env::var("THREADS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(4, |n| n.get()));

    let stop = Arc::new(AtomicBool::new(false));
    let scanned = Arc::new(AtomicU64::new(0));
    let winning_nonce = Arc::new(AtomicU64::new(0));

    let arc_ops = Arc::new(base_ops.to_vec());
    let arc_regs = Arc::new(regs);

    eprintln!(
        "Scanning nonces from {} to {} on {} threads...",
        start_nonce,
        start_nonce + max_scan,
        num_threads
    );

    let mut handles = Vec::new();
    let chunk_size = (max_scan + num_threads as u64 - 1) / num_threads as u64;

    let scan_start_time = Instant::now();

    for thread_idx in 0..num_threads {
        let t_start = start_nonce + (thread_idx as u64) * chunk_size;
        let t_end = (t_start + chunk_size).min(start_nonce + max_scan);

        let t_stop = Arc::clone(&stop);
        let t_scanned = Arc::clone(&scanned);
        let t_winning = Arc::clone(&winning_nonce);
        let t_ops = Arc::clone(&arc_ops);
        let t_regs = Arc::clone(&arc_regs);
        let t_hasher = base_hasher.clone();

        handles.push(std::thread::spawn(move || {
            let curve = secp256k1();
            for nonce in t_start..t_end {
                if t_stop.load(Ordering::Relaxed) {
                    break;
                }

                let is_clean = test_one_nonce(
                    &t_ops,
                    &t_hasher,
                    &t_regs,
                    total_qubits,
                    num_bits,
                    &curve,
                    nonce,
                );

                let c = t_scanned.fetch_add(1, Ordering::Relaxed);
                if c % 5000 == 0 && c > 0 {
                    let elapsed = scan_start_time.elapsed().as_secs_f64();
                    let rate = c as f64 / elapsed.max(0.001);
                    eprintln!("Scanned {} nonces ({:.1} nonces/sec)...", c, rate);
                }

                if is_clean {
                    eprintln!("*********************************************");
                    eprintln!("*** FOUND CLEAN NONCE: {} ***", nonce);
                    eprintln!("*********************************************");
                    t_winning.store(nonce, Ordering::SeqCst);
                    t_stop.store(true, Ordering::SeqCst);
                    let _ = std::fs::write("clean_nonce.txt", format!("{nonce}\n"));
                    break;
                }
            }
        }));
    }

    for h in handles {
        let _ = h.join();
    }

    let winner = winning_nonce.load(Ordering::SeqCst);
    if winner != 0 {
        eprintln!("SUCCESS! Found clean nonce: {winner}");
    } else {
        eprintln!(
            "Completed scanning {} nonces without finding a clean nonce.",
            scanned.load(Ordering::SeqCst)
        );
    }
}

fn test_one_nonce(
    base_ops: &[Op],
    base_hasher: &Shake256,
    layout_regs: &[Vec<QubitOrBit>],
    total_qubits: u64,
    num_bits: u64,
    curve: &WeierstrassEllipticCurve,
    nonce: u64,
) -> bool {
    let mut hasher = base_hasher.clone();
    // Append 96 tail X gates:
    for b in 0..48 {
        let t = (nonce >> b) & 1;
        for _ in 0..2 {
            hasher.update(&[OperationType::X as u8]);
            hasher.update(&NO_QUBIT.0.to_le_bytes());
            hasher.update(&NO_QUBIT.0.to_le_bytes());
            hasher.update(&t.to_le_bytes());
            hasher.update(&NO_BIT.0.to_le_bytes());
            hasher.update(&NO_BIT.0.to_le_bytes());
            hasher.update(&NO_REG.0.to_le_bytes());
        }
    }
    let mut xof = hasher.finalize_xof();
    const TARGET_SHOTS: usize = 9024;
    let mut raw_pairs = Vec::with_capacity(TARGET_SHOTS);
    for _ in 0..TARGET_SHOTS {
        let mut rb = [[0u8; 32]; 2];
        XofReader::read(&mut xof, &mut rb[0]);
        XofReader::read(&mut xof, &mut rb[1]);
        raw_pairs.push((U256::from_le_bytes(rb[0]), U256::from_le_bytes(rb[1])));
    }

    let mut sim = Simulator::new(total_qubits as usize, num_bits as usize, &mut xof);

    const BATCH: usize = 64;
    let mut raw_idx = 0;
    let mut total_valid_tested = 0;

    while raw_idx < raw_pairs.len() {
        let mut targets = Vec::with_capacity(BATCH);
        let mut offsets = Vec::with_capacity(BATCH);
        let mut expected = Vec::with_capacity(BATCH);

        while raw_idx < raw_pairs.len() && targets.len() < BATCH {
            let (k1, k2) = raw_pairs[raw_idx];
            raw_idx += 1;
            let t = curve.mul(curve.gx, curve.gy, k1);
            let o = curve.mul(curve.gx, curve.gy, k2);
            if t.0 == o.0 || (t.0.is_zero() && t.1.is_zero()) || (o.0.is_zero() && o.1.is_zero()) {
                continue;
            }
            let e = curve.add(t.0, t.1, o.0, o.1);
            targets.push(t);
            offsets.push(o);
            expected.push(e);
        }

        let bs = targets.len();
        if bs == 0 {
            break;
        }
        let cond_mask: u64 = if bs == 64 { u64::MAX } else { (1u64 << bs) - 1 };

        sim.clear_for_shot();
        for shot in 0..bs {
            sim.set_register(&layout_regs[0], targets[shot].0, shot);
            sim.set_register(&layout_regs[1], targets[shot].1, shot);
            sim.set_register(&layout_regs[2], offsets[shot].0, shot);
            sim.set_register(&layout_regs[3], offsets[shot].1, shot);
        }

        sim.apply_iter(base_ops.iter());

        // Check classical output:
        for shot in 0..bs {
            let gx = sim.get_register(&layout_regs[0], shot);
            let gy = sim.get_register(&layout_regs[1], shot);
            if gx != expected[shot].0 || gy != expected[shot].1 {
                return false; // Early reject on first classical failure
            }
        }

        // Check phase:
        let phase = sim.phase & cond_mask;
        if phase != 0 {
            return false; // Early reject on phase garbage
        }

        // Check ancilla:
        for register in layout_regs {
            for qb in register {
                if let QubitOrBit::Qubit(q) = *qb {
                    *sim.qubit_mut(q) = 0;
                }
            }
        }
        for q in 0..total_qubits {
            let v = sim.qubit(QubitId(q)) & cond_mask;
            if v != 0 {
                return false; // Early reject on dirty ancilla
            }
        }

        total_valid_tested += bs;
    }

    total_valid_tested > 0 // Passed all shots!
}
