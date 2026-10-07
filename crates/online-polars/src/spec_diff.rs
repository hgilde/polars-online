//! Where two lists of specs part, in words: what the refusal of a bank file
//! of other specs names (review 2026-10-06, PA6). "saved specs do not match
//! the bank's specs" alone left a resuming job of ten specs to diff them by
//! hand. Apart from `bank.rs`, which is held under 250 KB a source file
//! (`tests/test_repo_hygiene.py`).

use crate::spec::Spec;

/// One step of a path into a spec's JSON form.
#[derive(Debug, Clone)]
enum Seg {
    Key(String),
    Index(usize),
}

/// Where two lists of specs part: their count; or the first spec that
/// differs, by its name and place, and the first key of it whose value
/// differs, with both values. A key is named only if the file's value put
/// into the bank's spec changes that spec (`canon` fills and unshards it as
/// the comparison did): two spellings of one half-life, `"600s"` and
/// `"10m"`, are one spec and not a difference.
pub(crate) fn spec_difference(
    file: &[Spec],
    bank: &[Spec],
    canon: impl Fn(Spec) -> Spec,
) -> String {
    if file.len() != bank.len() {
        return format!(
            "the file holds {} specs and the bank {}",
            file.len(),
            bank.len()
        );
    }
    let n = file.len();
    let Some(i) = (0..n).find(|&i| file[i] != bank[i]) else {
        return "they differ".into();
    };
    let (f, b) = (&file[i], &bank[i]);
    if f.name != b.name {
        return format!(
            "spec {} of {n} is named {:?} in the file and {:?} in the bank",
            i + 1,
            f.name,
            b.name
        );
    }
    let place = format!("spec {:?} ({} of {n})", f.name, i + 1);
    let (Ok(fj), Ok(bj)) = (serde_json::to_value(f), serde_json::to_value(b)) else {
        return format!("{place} differs");
    };
    let mut leaves = Vec::new();
    differing_leaves(&fj, &bj, &mut Vec::new(), &mut leaves);
    for path in leaves {
        let mut probe = bj.clone();
        set_at(&mut probe, &path, value_at(&fj, &path).cloned());
        let same = serde_json::from_value::<Spec>(probe).is_ok_and(|s| canon(s) == *b);
        if !same {
            let shown =
                |v: Option<&serde_json::Value>| v.map_or("nothing".into(), ToString::to_string);
            let name = path
                .iter()
                .map(|s| match s {
                    Seg::Key(k) => format!(".{k}"),
                    Seg::Index(j) => format!("[{j}]"),
                })
                .collect::<String>();
            return format!(
                "{place} differs at {}: {} in the file, {} in the bank",
                name.trim_start_matches('.'),
                shown(value_at(&fj, &path)),
                shown(value_at(&bj, &path))
            );
        }
    }
    format!("{place} differs")
}

/// Every path at which `a` and `b` differ, leaves first in key order: a
/// key either lacks, a scalar, or an array of another length.
fn differing_leaves(
    a: &serde_json::Value,
    b: &serde_json::Value,
    at: &mut Vec<Seg>,
    out: &mut Vec<Vec<Seg>>,
) {
    use serde_json::Value;
    match (a, b) {
        (Value::Object(x), Value::Object(y)) => {
            let mut keys: Vec<&String> = x.keys().chain(y.keys()).collect();
            keys.sort();
            keys.dedup();
            for k in keys {
                at.push(Seg::Key(k.clone()));
                match (x.get(k), y.get(k)) {
                    (Some(va), Some(vb)) => differing_leaves(va, vb, at, out),
                    _ => out.push(at.clone()),
                }
                at.pop();
            }
        }
        (Value::Array(x), Value::Array(y)) if x.len() == y.len() => {
            for (j, (va, vb)) in x.iter().zip(y).enumerate() {
                at.push(Seg::Index(j));
                differing_leaves(va, vb, at, out);
                at.pop();
            }
        }
        _ if a == b => {}
        _ => out.push(at.clone()),
    }
}

fn value_at<'a>(v: &'a serde_json::Value, path: &[Seg]) -> Option<&'a serde_json::Value> {
    path.iter().try_fold(v, |v, s| match s {
        Seg::Key(k) => v.get(k.as_str()),
        Seg::Index(j) => v.get(*j),
    })
}

/// `v` with the value at `path` replaced by `new`, or removed for `None`.
fn set_at(v: &mut serde_json::Value, path: &[Seg], new: Option<serde_json::Value>) {
    let Some((last, parent)) = path.split_last() else {
        return;
    };
    let mut node = v;
    for s in parent {
        let next = match s {
            Seg::Key(k) => node.get_mut(k.as_str()),
            Seg::Index(j) => node.get_mut(*j),
        };
        let Some(next) = next else { return };
        node = next;
    }
    match (last, node, new) {
        (Seg::Key(k), serde_json::Value::Object(m), Some(x)) => {
            m.insert(k.clone(), x);
        }
        (Seg::Key(k), serde_json::Value::Object(m), None) => {
            m.remove(k);
        }
        (Seg::Index(j), serde_json::Value::Array(a), Some(x)) if *j < a.len() => a[*j] = x,
        _ => {}
    }
}
