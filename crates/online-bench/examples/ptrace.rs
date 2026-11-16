//! Run one budget case's setup and its measured block, for Processor Trace
//! (`PT_CASE=<id>`); ad-hoc signed with the get-task-allow entitlement, it is
//! what `xctrace record --template 'Processor Trace'` traces.
use online_bench::*;

fn main() {
    let id = std::env::var("PT_CASE").expect("PT_CASE");
    let mut sink = 0.0;
    if MODEL_CASES.contains(&id.as_str()) {
        let mut run = model_setup(&id);
        let b = polars_online_budget_model_rows(&mut run);
        sink += b.checksum;
    } else if BANK_CASES.contains(&id.as_str()) || WIDE_CASES.contains(&id.as_str()) {
        let mut run = bank_setup(&id);
        let out = polars_online_budget_bank_rows(&mut run);
        sink += out.len() as f64;
    } else {
        panic!("unknown case {id}");
    }
    println!("{id} {sink}");
}
