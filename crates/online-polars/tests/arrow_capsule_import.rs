//! The Arrow consumer obligation, discharged from outside polars-arrow.
//!
//! Importing a capsule makes the *consumer* take ownership: it must move the
//! C struct out and null the producer's `release` pointer. Skip that and the
//! producer's capsule destructor and the imported array's `Drop` are both
//! entitled to free the same buffers.
//!
//! `docs/PLAN.md` task 86 and an earlier `docs/ARROW-SOURCES.md` §4 recorded
//! this as a hard blocker, on the reasoning that polars-arrow keeps
//! `ArrowArray`'s fields `pub(super)` so the original cannot be marked
//! released. That was wrong, and this file is the correction. Marking it
//! released needs no field access: `ArrowArray::empty()` is public and builds
//! the struct with `release: None`, so one `std::ptr::replace` both moves the
//! producer's struct out and leaves a released struct behind. polars' own
//! `import_array_pycapsules` is this idiom.
//!
//! These run against the pinned `polars-arrow =0.55.2` that the wheel ships,
//! so they are a property of this build rather than of polars' main branch.
//! If a future polars-arrow drops `empty()`, they fail at build time instead
//! of at a double free.

use online_polars::export_struct_to_c;
use polars_arrow::array::{Array, Float64Array, StructArray};
use polars_arrow::datatypes::{ArrowDataType, Field as ArrowField};
use polars_arrow::ffi::{
    ArrowArray, export_array_to_c, export_field_to_c, import_array_from_c, import_field_from_c,
};

/// Whether the C struct's release callback is null, which is what "marked
/// released" means in the Arrow specification.
///
/// The fields are `pub(super)`, but `ArrowArray` derives `Debug` upstream, so
/// this reads the state without assuming a memory layout -- which is the whole
/// point, since the discarded fallback was to zero `release` through a raw
/// pointer at a guessed offset.
fn is_released(a: &ArrowArray) -> bool {
    format!("{a:?}").contains("release: None")
}

fn f64_array() -> Float64Array {
    Float64Array::from([Some(1.5), None, Some(-2.25), Some(0.0)])
}

/// The shape the bank actually exports: a struct of named f64 fields.
fn struct_array() -> StructArray {
    let fields = vec![
        ArrowField::new("pred_y".into(), ArrowDataType::Float64, true),
        ArrowField::new("n_eff".into(), ArrowDataType::Float64, true),
    ];
    let values: Vec<Box<dyn Array>> = vec![Box::new(f64_array()), Box::new(f64_array())];
    StructArray::new(ArrowDataType::Struct(fields), 4, values, None)
}

/// The control. A freshly exported struct carries a live callback, so the
/// assertion in the next test is measuring something that changed rather than
/// a state the struct was already in.
#[test]
fn a_freshly_exported_array_is_not_released() {
    let exported = export_array_to_c(Box::new(f64_array()) as Box<dyn Array>);
    assert!(
        !is_released(&exported),
        "a fresh export must carry a live release callback, or the release \
         assertion below proves nothing"
    );
}

#[test]
fn the_replace_idiom_marks_the_source_released_and_round_trips() {
    let field = ArrowField::new("x".into(), ArrowDataType::Float64, true);
    let schema = export_field_to_c(&field);
    let array = export_array_to_c(Box::new(f64_array()) as Box<dyn Array>);

    // The producer's struct, where a capsule would hold it.
    let mut produced = Box::new(array);
    let ptr: *mut ArrowArray = &mut *produced;
    assert!(!is_released(&produced));

    let dtype = unsafe { import_field_from_c(&schema) }.unwrap().dtype;

    // The consumer obligation, in one public call.
    let owned = unsafe { std::ptr::replace(ptr, ArrowArray::empty()) };
    let imported = unsafe { import_array_from_c(owned, dtype) }.unwrap();

    assert!(
        is_released(&produced),
        "the source must be marked released, or the producer's destructor and \
         the imported array both free the same buffers"
    );

    let got = imported.as_any().downcast_ref::<Float64Array>().unwrap();
    assert_eq!(got.len(), 4);
    assert_eq!(
        got.iter().collect::<Vec<_>>(),
        f64_array().iter().collect::<Vec<_>>()
    );

    // The emptied original must drop as a no-op, not a double free.
    drop(produced);
    drop(imported);
}

#[test]
fn the_struct_shape_the_bank_exports_round_trips_too() {
    let st = struct_array();
    let field = ArrowField::new("m".into(), st.dtype().clone(), true);
    let schema = export_field_to_c(&field);
    let array = export_array_to_c(Box::new(st) as Box<dyn Array>);

    let mut produced = Box::new(array);
    let ptr: *mut ArrowArray = &mut *produced;

    let imported_field = unsafe { import_field_from_c(&schema) }.unwrap();
    assert_eq!(imported_field.name.as_str(), "m");

    let owned = unsafe { std::ptr::replace(ptr, ArrowArray::empty()) };
    let imported = unsafe { import_array_from_c(owned, imported_field.dtype) }.unwrap();
    assert!(is_released(&produced));

    let got = imported.as_any().downcast_ref::<StructArray>().unwrap();
    assert_eq!(got.len(), 4);
    assert_eq!(got.values().len(), 2);
    drop(produced);
}

/// The shipping path, end to end. `export_struct_to_c` is what hands a spec's
/// output to Python, so importing *its* product is the round trip a real
/// consumer performs, rather than one assembled for the test.
#[test]
fn the_librarys_own_export_imports_back() {
    let (schema, array) = export_struct_to_c("m", struct_array());

    let mut produced = Box::new(array);
    let ptr: *mut ArrowArray = &mut *produced;
    assert!(!is_released(&produced));

    let field = unsafe { import_field_from_c(&schema) }.unwrap();
    assert_eq!(
        field.name.as_str(),
        "m",
        "the capsule carries the spec name"
    );

    let owned = unsafe { std::ptr::replace(ptr, ArrowArray::empty()) };
    let imported = unsafe { import_array_from_c(owned, field.dtype) }.unwrap();
    assert!(is_released(&produced));

    let got = imported.as_any().downcast_ref::<StructArray>().unwrap();
    assert_eq!(got.len(), 4);
    assert_eq!(got.values().len(), 2);
    drop(produced);
}

/// The schema half needs no replacement, which is why polars does not do one:
/// `import_field_from_c` takes it by reference and copies into a `Field`, so
/// the schema capsule's own destructor is still entitled to release it.
#[test]
fn the_schema_is_read_by_reference_and_stays_live() {
    let field = ArrowField::new("x".into(), ArrowDataType::Float64, true);
    let schema = export_field_to_c(&field);

    let first = unsafe { import_field_from_c(&schema) }.unwrap();
    let second = unsafe { import_field_from_c(&schema) }.unwrap();

    assert_eq!(first.name.as_str(), "x");
    assert_eq!(first.dtype, ArrowDataType::Float64);
    assert_eq!(
        first.name, second.name,
        "reading the schema must not consume it"
    );
}
