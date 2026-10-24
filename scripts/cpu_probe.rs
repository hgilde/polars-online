//! What a Rust program on this machine sees of its CPU, one `key=value` a
//! line, for `.github/workflows/cpu-paths.yml`.
//!
//!     rustc -O --edition 2024 scripts/cpu_probe.rs -o cpu_probe
//!     sde -hsw -- ./cpu_probe
//!
//! The product kernels choose their code and their blocking at run time
//! from the same two sources: the CPU's features, through
//! `is_x86_feature_detected!`, and its caches, through the CPUID
//! instruction (on Windows `private-gemm-x86` reads leaf 4, the
//! deterministic cache parameters, on an Intel CPU, and leaf `0x8000_0006`
//! on an AMD one). Run under Intel SDE, this prints what SDE made the
//! program see, so the workflow can show that the chip it asked for is the
//! one the tests ran on, before it believes a green run.
//!
//! Not part of the workspace; one file, standard library only, built with
//! `rustc` in the job that runs it.

#[cfg(target_arch = "x86_64")]
fn main() {
    use std::arch::x86_64::__cpuid_count;

    let leaf0 = __cpuid_count(0, 0);
    let vendor: Vec<u8> = [leaf0.ebx, leaf0.edx, leaf0.ecx]
        .iter()
        .flat_map(|r| r.to_le_bytes())
        .collect();
    let vendor = String::from_utf8_lossy(&vendor).into_owned();
    println!("vendor={vendor}");

    let top_ext = __cpuid_count(0x8000_0000, 0).eax;
    if top_ext >= 0x8000_0004 {
        let brand: Vec<u8> = (0x8000_0002u32..=0x8000_0004)
            .flat_map(|leaf| {
                let r = __cpuid_count(leaf, 0);
                [r.eax, r.ebx, r.ecx, r.edx]
            })
            .flat_map(|r| r.to_le_bytes())
            .collect();
        let brand = String::from_utf8_lossy(&brand);
        println!("brand={}", brand.trim_matches(char::from(0)).trim());
    }

    println!("sse3={}", is_x86_feature_detected!("sse3"));
    println!("avx={}", is_x86_feature_detected!("avx"));
    println!("avx2={}", is_x86_feature_detected!("avx2"));
    println!("fma={}", is_x86_feature_detected!("fma"));
    println!("avx512f={}", is_x86_feature_detected!("avx512f"));

    // Leaf 4, one subleaf a cache until a null type: the size is ways x
    // partitions x line x sets, each field stored less one.
    let mut l3_seen = false;
    if leaf0.eax >= 4 {
        for sub in 0..16 {
            let r = __cpuid_count(4, sub);
            let kind = r.eax & 0x1f;
            if kind == 0 {
                break;
            }
            if kind == 2 {
                // An instruction cache, which no product reads.
                continue;
            }
            let level = (r.eax >> 5) & 0x7;
            let ways = u64::from((r.ebx >> 22) + 1);
            let partitions = u64::from(((r.ebx >> 12) & 0x3ff) + 1);
            let line = u64::from((r.ebx & 0xfff) + 1);
            let sets = u64::from(r.ecx) + 1;
            let kib = ways * partitions * line * sets / 1024;
            println!("leaf4_l{level}_kib={kib}");
            if level == 3 {
                println!("l3_kib={kib}");
                l3_seen = true;
            }
        }
    }
    // AMD's: L2 in ECX[31:16] KiB, L3 in EDX[31:18] units of 512 KiB.
    if top_ext >= 0x8000_0006 {
        let r = __cpuid_count(0x8000_0006, 0);
        println!("leaf80000006_l2_kib={}", r.ecx >> 16);
        let l3 = u64::from(r.edx >> 18) * 512;
        println!("leaf80000006_l3_kib={l3}");
        if !l3_seen && vendor == "AuthenticAMD" {
            println!("l3_kib={l3}");
        }
    }
}

#[cfg(not(target_arch = "x86_64"))]
fn main() {
    println!("arch={}", std::env::consts::ARCH);
}
