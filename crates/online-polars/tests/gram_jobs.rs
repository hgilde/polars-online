//! `gramfit::fit_jobs` (docs/PLAN.md task 256): a fit per job, each its own
//! subset, target, draw of signs and columns. Each job is held, to the bit,
//! to its columns merged alone -- `merge` or `merge_within` on the
//! intercept and the job's columns, the draw's signs through
//! `within_cross` -- and fitted by `lars_paths` or `ridge_fits` on that.
//! The pool's size moves no bit, and every index out of range is refused.

use online_polars::Threads;
use online_polars::gramfit::{
    ColumnSet, FitColumns, GramPath, JobFit, Pooling, SubsetFit, SubsetJob, SubsetOut, fit_jobs,
    fit_subsets, lars_paths,
};
use online_polars::online_core::gramfit::{
    GramArrays, LarsLimits, OwnedGram, merge, merge_within, ridge_fits, within_cross,
};

const K: usize = 9; // the intercept and eight features
const M: usize = 2;

/// A deterministic stream of numbers in about `[-1, 1]`.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }
}

/// The Gram of `rows` rows: the intercept at column 0, features at a level
/// of their own sharing a factor, two targets on some of them.
fn gram(seed: u64, rows: usize) -> OwnedGram {
    let mut r = Lcg(seed);
    let level: Vec<f64> = (0..K).map(|_| 2.0 * r.next()).collect();
    let mut z = Vec::with_capacity(rows * K);
    let mut y = Vec::with_capacity(rows * M);
    for _ in 0..rows {
        let f = r.next();
        let row: Vec<f64> = (0..K)
            .map(|c| {
                if c == 0 {
                    1.0
                } else {
                    r.next() + 0.6 * f + level[c]
                }
            })
            .collect();
        y.push(0.6 * row[1] - 0.4 * row[3] + 0.3 * row[6] + 0.5 * r.next());
        y.push(0.5 * row[2] + 0.5 * row[5] - 0.3 * row[8] + 0.5 * r.next());
        z.extend(row);
    }
    let n = rows as f64;
    let means: Vec<f64> = (0..K)
        .map(|c| (0..rows).map(|i| z[i * K + c]).sum::<f64>() / n)
        .collect();
    let dz = |i: usize, c: usize| z[i * K + c] - means[c];
    let comoments: Vec<f64> = (0..K * K)
        .map(|e| (0..rows).map(|i| dz(i, e / K) * dz(i, e % K)).sum::<f64>() / n)
        .collect();
    let ybar: Vec<f64> = (0..M)
        .map(|t| (0..rows).map(|i| y[i * M + t]).sum::<f64>() / n)
        .collect();
    let cross_moments: Vec<f64> = (0..M * K)
        .map(|e| {
            (0..rows)
                .map(|i| z[i * K + e % K] * y[i * M + e / K])
                .sum::<f64>()
                / n
        })
        .collect();
    let cross_centred: Vec<f64> = (0..M * K)
        .map(|e| {
            let (t, c) = (e / K, e % K);
            (0..rows)
                .map(|i| dz(i, c) * (y[i * M + t] - ybar[t]))
                .sum::<f64>()
                / n
        })
        .collect();
    let target_vars = (0..M)
        .map(|t| {
            (0..rows)
                .map(|i| (y[i * M + t] - ybar[t]).powi(2))
                .sum::<f64>()
                / n
        })
        .collect();
    OwnedGram {
        k: K,
        weight_sum: n,
        means_by_target: means.repeat(M),
        means,
        comoments,
        cross_moments,
        cross_centred,
        target_weights: vec![n; M],
        target_means: ybar,
        target_vars,
        target_n_kish: vec![n; M],
        within: None,
    }
}

fn grams() -> Vec<OwnedGram> {
    (0..5).map(|i| gram(17 + i, 60 + 7 * i as usize)).collect()
}

fn subsets() -> Vec<Vec<usize>> {
    vec![vec![0, 1, 2], vec![3, 4], vec![4, 0, 2, 1], vec![2]]
}

/// Two draws: every sign +1, and some -1.
fn signs() -> Vec<Vec<f64>> {
    vec![vec![1.0; 5], vec![1.0, -1.0, -1.0, 1.0, -1.0]]
}

/// Column sets of their own order and size, one weighted.
fn sets() -> Vec<ColumnSet> {
    let set = |slots: &[usize]| ColumnSet {
        slots: slots.to_vec(),
        weights: Vec::new(),
    };
    vec![
        set(&[3, 1, 6]),
        set(&[8, 2, 5, 7, 1]),
        set(&[1, 2, 3, 4, 5, 6, 7, 8]),
        ColumnSet {
            slots: vec![6, 3, 2, 1],
            weights: vec![0.5, 1.0, 2.0, 1.5],
        },
        set(&[4]),
    ]
}

/// A job for every subset, draw, target and set, in an order that is not
/// the nested one.
fn jobs(n_draws: usize) -> Vec<SubsetJob> {
    let mut out = Vec::new();
    for set in 0..sets().len() {
        for draw in (0..n_draws).rev() {
            for subset in 0..subsets().len() {
                for target in 0..M {
                    out.push(SubsetJob {
                        subset,
                        target,
                        draw,
                        set,
                    });
                }
            }
        }
    }
    out
}

fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn pool(n: usize) -> Threads {
    Threads::new(Some(n)).unwrap()
}

/// `job`'s columns merged alone: the intercept and the set's slots, the
/// draw's signs on the cross-moments under `within`.
fn alone(
    views: &[GramArrays<'_>],
    job: SubsetJob,
    within: bool,
    draws: &[Vec<f64>],
) -> (OwnedGram, Vec<usize>) {
    let subset = &subsets()[job.subset];
    let parts: Vec<GramArrays<'_>> = subset.iter().map(|&i| views[i]).collect();
    let cols: Vec<usize> = std::iter::once(0)
        .chain(sets()[job.set].slots.iter().copied())
        .collect();
    if !within {
        return (merge(&parts, &cols, Some(0)), cols);
    }
    let mut g = merge_within(&parts, &cols, Some(0));
    if let Some(draw) = draws.get(job.draw) {
        let own: Vec<f64> = subset.iter().map(|&i| draw[i]).collect();
        let cc = within_cross(&parts, &cols, &own);
        g.cross_moments.clone_from(&cc);
        g.cross_centred = cc;
    }
    (g, cols)
}

/// A path over a merge's columns `cols` as one over the Grams' `K`.
fn widened(p: &GramPath, cols: &[usize]) -> (Vec<f64>, Vec<Vec<usize>>) {
    let n = cols.len();
    let mut coefs = vec![0.0; p.penalties.len() * K];
    for r in 0..p.penalties.len() {
        for (q, &c) in cols.iter().enumerate() {
            coefs[r * K + c] = p.coefs[r * n + q];
        }
    }
    let active = p
        .active
        .iter()
        .map(|a| a.iter().map(|&q| cols[q]).collect())
        .collect();
    (coefs, active)
}

fn path_case(within: bool, draws: Vec<Vec<f64>>, limits: LarsLimits) {
    let owned = grams();
    let views: Vec<GramArrays<'_>> = owned.iter().map(OwnedGram::arrays).collect();
    let n_draws = draws.len().max(1);
    let jobs = jobs(n_draws);
    let pooling = if within {
        Pooling::Within {
            signs: draws.clone(),
        }
    } else {
        Pooling::Union
    };
    let got = fit_jobs(
        &views,
        &subsets(),
        &sets(),
        &jobs,
        Some(0),
        &JobFit::Path { limits },
        &pooling,
        &pool(3),
    )
    .unwrap();
    assert_eq!(got.len(), jobs.len());
    let mut knots = 0;
    for (job, out) in jobs.iter().zip(&got) {
        let SubsetOut::Path(path) = out else {
            panic!("a path job gave a ridge fit")
        };
        let (g, cols) = alone(&views, *job, within, &draws);
        let set = &sets()[job.set];
        let fit_cols = FitColumns {
            slots: (1..cols.len()).collect(),
            icept: Some(0),
        };
        let want = &lars_paths(
            &[g.arrays()],
            &[vec![job.target]],
            &fit_cols,
            &set.weights,
            limits,
            &pool(1),
        )
        .unwrap()[0][0];
        let (coefs, active) = widened(want, &cols);
        assert_eq!(bits(&path.penalties), bits(&want.penalties), "{job:?}");
        assert_eq!(bits(&path.coefs), bits(&coefs), "{job:?}");
        assert_eq!(path.active, active, "{job:?}");
        assert_eq!(path.stop, want.stop, "{job:?}");
        knots += path.penalties.len();
    }
    // The paths ran: more than a knot each on average.
    assert!(knots > 2 * jobs.len(), "{knots} knots");
}

#[test]
fn each_path_is_its_columns_merged_within_alone_under_its_draw() {
    let limits = LarsLimits {
        max_steps: None,
        max_active: Some(3),
    };
    path_case(true, signs(), limits);
}

#[test]
fn each_path_is_its_columns_merged_alone_without_signs() {
    path_case(true, Vec::new(), LarsLimits::default());
    path_case(false, Vec::new(), LarsLimits::default());
}

#[test]
fn a_draw_of_signs_moves_the_paths() {
    let owned = grams();
    let views: Vec<GramArrays<'_>> = owned.iter().map(OwnedGram::arrays).collect();
    let jobs: Vec<SubsetJob> = (0..2)
        .map(|draw| SubsetJob {
            subset: 0,
            target: 0,
            draw,
            set: 2,
        })
        .collect();
    let out = fit_jobs(
        &views,
        &subsets(),
        &sets(),
        &jobs,
        Some(0),
        &JobFit::Path {
            limits: LarsLimits::default(),
        },
        &Pooling::Within { signs: signs() },
        &pool(2),
    )
    .unwrap();
    let [SubsetOut::Path(a), SubsetOut::Path(b)] = &out[..] else {
        panic!("two paths")
    };
    assert_ne!(bits(&a.coefs), bits(&b.coefs));
}

#[test]
fn each_ridge_is_its_columns_merged_alone() {
    let owned = grams();
    let views: Vec<GramArrays<'_>> = owned.iter().map(OwnedGram::arrays).collect();
    let draws = signs();
    let jobs = jobs(draws.len());
    for (ridge, standardize) in [(0.0, false), (0.3, true)] {
        let got = fit_jobs(
            &views,
            &subsets(),
            &sets(),
            &jobs,
            Some(0),
            &JobFit::Ridge { ridge, standardize },
            &Pooling::Within {
                signs: draws.clone(),
            },
            &pool(4),
        )
        .unwrap();
        for (job, out) in jobs.iter().zip(&got) {
            let SubsetOut::Ridge(fit) = out else {
                panic!("a ridge job gave a path")
            };
            let (g, cols) = alone(&views, *job, true, &draws);
            let slots: Vec<usize> = (1..cols.len()).collect();
            let want = &ridge_fits(
                &g.arrays(),
                &[job.target],
                &slots,
                Some(0),
                ridge,
                standardize,
            )[0];
            let wide = |v: &[f64], fill: f64| {
                let mut out = vec![fill; K];
                for (q, &c) in cols.iter().enumerate() {
                    out[c] = v[q];
                }
                out
            };
            assert_eq!(bits(&fit.coef), bits(&wide(&want.coef, 0.0)), "{job:?}");
            assert_eq!(bits(&fit.se), bits(&wide(&want.se, f64::NAN)), "{job:?}");
            assert_eq!(bits(&fit.t), bits(&wide(&want.t, f64::NAN)), "{job:?}");
            for (a, b) in [
                (fit.resid_var, want.resid_var),
                (fit.sigma2, want.sigma2),
                (fit.r2, want.r2),
                (fit.n, want.n),
            ] {
                assert_eq!(a.to_bits(), b.to_bits(), "{job:?}");
            }
            assert!(fit.coef[1..].iter().any(|&b| b != 0.0));
        }
    }
}

#[test]
fn fit_subsets_is_a_job_per_draw_subset_and_target() {
    let owned = grams();
    let views: Vec<GramArrays<'_>> = owned.iter().map(OwnedGram::arrays).collect();
    let set = &sets()[3];
    let cols = FitColumns {
        slots: set.slots.clone(),
        icept: Some(0),
    };
    let limits = LarsLimits {
        max_steps: Some(3),
        max_active: None,
    };
    let fit = SubsetFit::Path {
        weights: set.weights.clone(),
        limits,
    };
    let pooling = Pooling::Within { signs: signs() };
    let nested = fit_subsets(&views, &subsets(), &[1, 0], &cols, &fit, &pooling, &pool(2)).unwrap();
    assert_eq!(nested.len(), 2);
    for (draw, per) in nested.iter().enumerate() {
        assert_eq!(per.len(), subsets().len());
        for (subset, fits) in per.iter().enumerate() {
            for (fit, target) in fits.iter().zip([1, 0]) {
                let job = [SubsetJob {
                    subset,
                    target,
                    draw,
                    set: 0,
                }];
                let want = fit_jobs(
                    &views,
                    &subsets(),
                    std::slice::from_ref(set),
                    &job,
                    Some(0),
                    &JobFit::Path { limits },
                    &pooling,
                    &pool(1),
                )
                .unwrap();
                let (SubsetOut::Path(a), SubsetOut::Path(b)) = (fit, &want[0]) else {
                    panic!("two paths")
                };
                // Bits, since the intercept is nan within the parts.
                assert_eq!(bits(&a.coefs), bits(&b.coefs));
                assert_eq!(bits(&a.penalties), bits(&b.penalties));
                assert_eq!((&a.active, a.stop), (&b.active, b.stop));
            }
        }
    }
}

#[test]
fn the_pools_size_moves_no_bit() {
    let owned = grams();
    let views: Vec<GramArrays<'_>> = owned.iter().map(OwnedGram::arrays).collect();
    let jobs = jobs(2);
    let run = |n: usize, fit: JobFit| {
        fit_jobs(
            &views,
            &subsets(),
            &sets(),
            &jobs,
            Some(0),
            &fit,
            &Pooling::Within { signs: signs() },
            &pool(n),
        )
        .unwrap()
    };
    for fit in [
        JobFit::Path {
            limits: LarsLimits::default(),
        },
        JobFit::Ridge {
            ridge: 0.1,
            standardize: false,
        },
    ] {
        let (one, four) = (run(1, fit), run(4, fit));
        for (a, b) in one.iter().zip(&four) {
            match (a, b) {
                (SubsetOut::Path(a), SubsetOut::Path(b)) => {
                    assert_eq!(bits(&a.coefs), bits(&b.coefs));
                    assert_eq!(bits(&a.penalties), bits(&b.penalties));
                    assert_eq!(a.active, b.active);
                }
                (SubsetOut::Ridge(a), SubsetOut::Ridge(b)) => {
                    assert_eq!(bits(&a.coef), bits(&b.coef));
                    assert_eq!(bits(&a.se), bits(&b.se));
                }
                _ => panic!("a fit of another kind"),
            }
        }
    }
}

#[test]
fn no_jobs_fit_nothing_and_merge_nothing() {
    let owned = grams();
    let views: Vec<GramArrays<'_>> = owned.iter().map(OwnedGram::arrays).collect();
    let out = fit_jobs(
        &views,
        &subsets(),
        &sets(),
        &[],
        Some(0),
        &JobFit::Path {
            limits: LarsLimits::default(),
        },
        &Pooling::Union,
        &pool(1),
    )
    .unwrap();
    assert!(out.is_empty());
}

#[test]
fn every_index_out_of_range_is_refused_by_name() {
    let owned = grams();
    let views: Vec<GramArrays<'_>> = owned.iter().map(OwnedGram::arrays).collect();
    let ok = SubsetJob {
        subset: 1,
        target: 1,
        draw: 1,
        set: 1,
    };
    let other = gram(99, 30);
    let mut short = other.clone();
    short.k = K - 1;
    short.means.pop();
    let mut fewer = other.clone();
    fewer.target_weights.pop();
    let set = |slots: Vec<usize>, weights: Vec<f64>| vec![ColumnSet { slots, weights }];
    type Case = (
        &'static str,
        Vec<SubsetJob>,
        Vec<Vec<usize>>,
        Vec<ColumnSet>,
        Vec<Vec<f64>>,
        Option<OwnedGram>,
    );
    let cases: Vec<Case> = vec![
        (
            "job 0 names subset 4 of 4",
            vec![SubsetJob { subset: 4, ..ok }],
            subsets(),
            sets(),
            signs(),
            None,
        ),
        (
            "job 1 names column set 5 of 5",
            vec![ok, SubsetJob { set: 5, ..ok }],
            subsets(),
            sets(),
            signs(),
            None,
        ),
        (
            "job 0 names draw 2 of 2",
            vec![SubsetJob { draw: 2, ..ok }],
            subsets(),
            sets(),
            signs(),
            None,
        ),
        (
            "job 0 names draw 1 of 1",
            vec![ok],
            subsets(),
            sets(),
            Vec::new(),
            None,
        ),
        (
            "job 0 names target 2 of 2",
            vec![SubsetJob { target: 2, ..ok }],
            subsets(),
            sets(),
            signs(),
            None,
        ),
        (
            "subset 1 names no Gram",
            vec![ok],
            vec![vec![0], vec![]],
            sets(),
            signs(),
            None,
        ),
        (
            "subset 0 names Gram 5 of 5",
            vec![ok],
            vec![vec![5], vec![1]],
            sets(),
            signs(),
            None,
        ),
        (
            "a draw has 4 signs for 5 Grams",
            vec![ok],
            subsets(),
            sets(),
            vec![vec![1.0; 5], vec![1.0; 4]],
            None,
        ),
        (
            "column set 0 names column 9",
            vec![SubsetJob { set: 0, ..ok }],
            subsets(),
            set(vec![1, 9], Vec::new()),
            signs(),
            None,
        ),
        (
            "column set 0 names column 0",
            vec![SubsetJob { set: 0, ..ok }],
            subsets(),
            set(vec![0, 2], Vec::new()),
            signs(),
            None,
        ),
        // Task 269, D-3: a column twice is one regressor solved twice, its
        // coefficient written once, the last write winning.
        (
            "column set 0 names column 2 twice",
            vec![SubsetJob { set: 0, ..ok }],
            subsets(),
            set(vec![1, 2, 3, 2], Vec::new()),
            signs(),
            None,
        ),
        (
            "column set 0: 1 penalty weights for 2 features",
            vec![SubsetJob { set: 0, ..ok }],
            subsets(),
            set(vec![1, 2], vec![1.0]),
            signs(),
            None,
        ),
        (
            "every Gram must have the same columns and targets: 9 and 2, then 8 and 2",
            vec![ok],
            subsets(),
            sets(),
            signs(),
            Some(short),
        ),
        (
            "every Gram must have the same columns and targets: 9 and 2, then 9 and 1",
            vec![ok],
            subsets(),
            sets(),
            signs(),
            Some(fewer),
        ),
    ];
    for (want, jobs, subsets, sets, draws, extra) in cases {
        let mut views = views.clone();
        if let Some(g) = &extra {
            views.push(g.arrays());
        }
        let pooling = Pooling::Within { signs: draws };
        for fit in [
            JobFit::Path {
                limits: LarsLimits::default(),
            },
            JobFit::Ridge {
                ridge: 0.0,
                standardize: false,
            },
        ] {
            let err = fit_jobs(
                &views,
                &subsets,
                &sets,
                &jobs,
                Some(0),
                &fit,
                &pooling,
                &pool(1),
            )
            .unwrap_err()
            .to_string();
            assert!(err.contains(want), "{err:?} should name {want:?}");
        }
    }
    // The intercept itself out of range.
    let err = fit_jobs(
        &views,
        &subsets(),
        &sets(),
        &[ok],
        Some(K),
        &JobFit::Path {
            limits: LarsLimits::default(),
        },
        &Pooling::Within { signs: signs() },
        &pool(1),
    )
    .unwrap_err()
    .to_string();
    assert!(
        err.contains("intercept column 9 out of range for 9 columns"),
        "{err}"
    );
}
