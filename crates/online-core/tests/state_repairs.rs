//! A state of this build's layout is what loads: a field written since some
//! older layout is required, and a vector of the wrong length is refused, as
//! every other vector of a state is (docs/PLAN.md task 198; review round 4,
//! CC8).
//!
//! Before 1.0, a file older than `SCHEMA_VERSION` is refused by its version
//! (`MIN_SCHEMA_VERSION`), so a repair for a layout written before a field
//! loaded nothing a file this build reads can hold: `robust`'s row counts
//! from its weights, `holt`'s clock since an observation at zero, the update
//! models' target weights from the shared one, `ftrl`'s penalty scale at 1,
//! and every mean's low part (`comp.rs`) at zero. What each did reach was a
//! damaged file, whose vector of the wrong length it mended with numbers the
//! fit never had. From 1.0 a layout change ships a loader of its own, held
//! to a frozen fixture (`tests/state_fixtures.rs`), not a default.

mod cases;

use std::panic::{AssertUnwindSafe, catch_unwind};

use cases::{Case, all, named, saved};
use rmpv::Value;

/// A path into a msgpack value: map keys and array indices.
#[derive(Debug, Clone)]
enum Step {
    Key(String),
    Index(usize),
}

fn key_of(k: &Value) -> Option<&str> {
    k.as_str()
}

fn at<'a>(v: &'a mut Value, path: &[Step]) -> &'a mut Value {
    let mut node = v;
    for s in path {
        node = match (s, node) {
            (Step::Key(name), Value::Map(entries)) => {
                &mut entries
                    .iter_mut()
                    .find(|(k, _)| key_of(k) == Some(name))
                    .unwrap_or_else(|| panic!("no key {name}"))
                    .1
            }
            (Step::Index(i), Value::Array(items)) => &mut items[*i],
            (s, other) => panic!("{s:?} into {other}"),
        };
    }
    node
}

/// Every path to a map entry whose key `pick` accepts.
fn paths(v: &Value, here: &mut Vec<Step>, pick: &dyn Fn(&str) -> bool, out: &mut Vec<Vec<Step>>) {
    match v {
        Value::Map(entries) => {
            for (k, val) in entries {
                let Some(name) = key_of(k) else { continue };
                here.push(Step::Key(name.to_string()));
                if pick(name) {
                    out.push(here.clone());
                }
                paths(val, here, pick, out);
                here.pop();
            }
        }
        Value::Array(items) => {
            for (i, val) in items.iter().enumerate() {
                here.push(Step::Index(i));
                paths(val, here, pick, out);
                here.pop();
            }
        }
        _ => {}
    }
}

/// The entry at `path` taken out of its map.
fn removed(v: &Value, path: &[Step]) -> Value {
    let mut v = v.clone();
    let (last, parent) = path.split_last().unwrap();
    let Step::Key(name) = last else {
        unreachable!()
    };
    let Value::Map(entries) = at(&mut v, parent) else {
        unreachable!()
    };
    entries.retain(|(k, _)| key_of(k) != Some(name));
    v
}

/// The vector at `path` with `edit` applied, or `None` where the edit would
/// change nothing.
/// An edit of a vector: whether it changed anything.
type Edit = fn(&mut Vec<Value>) -> bool;

fn edited(v: &Value, path: &[Step], edit: Edit) -> Option<Value> {
    let mut v = v.clone();
    let Value::Array(items) = at(&mut v, path) else {
        return None;
    };
    edit(items).then_some(v)
}

fn emptied(items: &mut Vec<Value>) -> bool {
    let changed = !items.is_empty();
    items.clear();
    changed
}

fn one_short(items: &mut Vec<Value>) -> bool {
    items.pop().is_some()
}

/// The first nested vector one short, for a vector of vectors.
#[allow(clippy::ptr_arg)] // one signature for every edit
fn inner_one_short(items: &mut Vec<Value>) -> bool {
    match items.first_mut() {
        Some(Value::Array(inner)) => inner.pop().is_some(),
        _ => false,
    }
}

fn encode(v: &Value) -> Vec<u8> {
    let mut out = Vec::new();
    rmpv::encode::write_value(&mut out, v).unwrap();
    out
}

/// What became of a damaged state: refused at the load, as it must be, or
/// loaded -- and then either ran the case's continuation or panicked in it.
fn fate(case: &Case, bytes: &[u8]) -> Result<String, String> {
    let rows = case.rows();
    match catch_unwind(AssertUnwindSafe(|| cases::load(case, bytes))) {
        Err(_) => Ok("panicked while loading".into()),
        Ok(Err(e)) => Err(e),
        Ok(Ok(mut m)) => {
            let ran = catch_unwind(AssertUnwindSafe(|| {
                for r in &rows[case.before..] {
                    m.step(r);
                }
            }));
            Ok(match ran {
                Ok(()) => "loaded and ran on".into(),
                Err(_) => "loaded, then panicked on the next rows".into(),
            })
        }
    }
}

fn describe(path: &[Step]) -> String {
    path.iter()
        .map(|s| match s {
            Step::Key(k) => k.clone(),
            Step::Index(i) => i.to_string(),
        })
        .collect::<Vec<_>>()
        .join(".")
}

/// Each damaged form of `v` at `path`: the entry taken out, and each edit of
/// it that changes something.
fn damaged(v: &Value, path: &[Step]) -> Vec<(&'static str, Value)> {
    let mut out = vec![("left out", removed(v, path))];
    let edits: [(&str, Edit); 3] = [
        ("emptied", emptied),
        ("one short", one_short),
        ("an inner vector one short", inner_one_short),
    ];
    for (how, edit) in edits {
        if let Some(e) = edited(v, path, edit) {
            out.push((how, e));
        }
    }
    out
}

fn state_value(case: &Case) -> Value {
    let (m, _) = saved(case);
    rmpv::decode::read_value(&mut named(&m.state()).as_slice()).unwrap()
}

/// Every mean's low part, wherever a state keeps one -- a model's own, an
/// accumulator's, a window's snapshot's -- is refused left out, emptied or
/// short. A low part was sized at its first use, so a state written before
/// it loaded, and so did a damaged one, its means then off by what their
/// low parts held.
#[test]
fn a_low_part_of_the_wrong_length_is_refused() {
    let mut loaded = Vec::new();
    let mut seen = 0;
    for case in all() {
        let v = state_value(&case);
        let mut found = Vec::new();
        paths(&v, &mut Vec::new(), &|k| k.ends_with("_lo"), &mut found);
        for path in found {
            for (how, bad) in damaged(&v, &path) {
                seen += 1;
                if let Ok(what) = fate(&case, &encode(&bad)) {
                    loaded.push(format!("{}: {} {how}: {what}", case.name, describe(&path)));
                }
            }
        }
    }
    assert!(seen > 100, "the cases keep low parts to damage: {seen}");
    assert!(
        loaded.is_empty(),
        "{} of {seen} damaged low parts were not refused:\n{}",
        loaded.len(),
        loaded.join("\n")
    );
}

/// The fields review round 4 found repaired for a layout written before
/// them (CC8), each refused left out, emptied or short, where each loaded
/// as the repair made it: `robust`'s row counts as its weights, `holt`'s
/// clock since an observation at zero, the update models' target weights at
/// the shared weight, `ftrl`'s owed decay at none (with the penalty scale
/// task 215 removed).
#[test]
fn a_field_written_since_an_older_layout_is_required() {
    const FIELDS: [(&str, &str, &str); 7] = [
        ("huber", "Robust", "nobs"),
        ("holt", "Holt", "since"),
        ("sgd", "Sgd", "w_target"),
        ("pa", "Pa", "w_target"),
        ("rls", "Rls", "w_target"),
        ("ftrl", "Ftrl", "w_target"),
        ("ftrl", "Ftrl", "pending"),
    ];
    let cases = all();
    let mut loaded = Vec::new();
    for (name, variant, field) in FIELDS {
        let case = cases.iter().find(|c| c.name == name).unwrap();
        let v = state_value(case);
        let path = [
            Step::Key("model".into()),
            Step::Key(variant.into()),
            Step::Key(field.into()),
        ];
        for (how, bad) in damaged(&v, &path) {
            if let Ok(what) = fate(case, &encode(&bad)) {
                loaded.push(format!("{name}: {field} {how}: {what}"));
            }
        }
    }
    assert!(
        loaded.is_empty(),
        "{} damaged states were not refused:\n{}",
        loaded.len(),
        loaded.join("\n")
    );
}
