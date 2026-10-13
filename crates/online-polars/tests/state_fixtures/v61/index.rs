// The frozen fixtures of schema 61, kept when schema 62 moved `ewridge`'s
// layout (docs/PLAN.md task 257): the bank files and the `refresh_time`
// states, which the promise covers, as `tests/state_fixtures/index.rs`
// listed them at 61. Not the windows states, which are labelled unstable
// and regenerated with their changes.

pub mod bank {
    include!("bank.rs");
}
pub mod bank_diagnosed {
    include!("bank_diagnosed.rs");
}
pub mod refresh_time {
    include!("refresh_time.rs");
}
pub mod refresh_time_int {
    include!("refresh_time_int.rs");
}

/// Every fixture.
pub const ALL: &[&crate::Fixture] = &[
    &bank::FIXTURE,
    &bank_diagnosed::FIXTURE,
    &refresh_time::FIXTURE,
    &refresh_time_int::FIXTURE,
];
