//! Arrow IPC files read record batch by record batch, with no query engine
//! between the file and the library: what the research measured fastest and
//! leanest (docs/GBDT-ADAPTER.md §4.1, option D).
//!
//! Polars' streaming engine reads ahead of its consumer by many row groups
//! or files (`POLARS_ROW_GROUP_PREFETCH_SIZE`, `POLARS_MAX_CONCURRENT_SCANS`;
//! on 10 million rows of 50 features `collect_batches` held 2.1 GB at once,
//! and on a 1 GB source all of it, task 242). This reader holds one record
//! batch, and the build's reader thread a few more.
//!
//! Each file's columns are found by name, in the [`Layout`]'s order, and
//! converted where a library needs it, as the Python layer's plan converts a
//! polars frame's: a Boolean feature to `UInt8`, a dictionary column (how
//! polars writes a `pl.Enum`) to its codes in the Enum's categories, the
//! label and weight to `Float64`. A dictionary that is the Enum's categories
//! in order, as polars writes one, hands over its keys as they are.

use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;

use polars_arrow::array::{
    Array, BooleanArray, DictionaryArray, DictionaryKey, PrimitiveArray, Utf8Array, Utf8ViewArray,
};
use polars_arrow::datatypes::{ArrowDataType, IntegerType};
use polars_arrow::io::ipc::read::{FileReader, read_file_metadata};

use super::frames::{Frame, Frames, Num, column_name};
use super::{GbdtError, Kind, Layout};

/// Arrow IPC files, read in order, the columns of [`Layout`] taken from each.
pub struct IpcFrames {
    files: Vec<PathBuf>,
    layout: Layout,
    /// Each feature's Enum categories, for a dictionary column.
    categories: Vec<Option<Vec<String>>>,
}

impl IpcFrames {
    /// `categories[j]` is feature `j`'s Enum categories when it is one.
    pub fn new(
        files: Vec<PathBuf>,
        layout: Layout,
        categories: Vec<Option<Vec<String>>>,
    ) -> Result<Self, GbdtError> {
        if files.is_empty() {
            return Err(GbdtError::msg("no Arrow IPC files"));
        }
        if categories.len() != layout.names.len() {
            return Err(GbdtError::msg("a category list per feature"));
        }
        Ok(Self {
            files,
            layout,
            categories,
        })
    }

    fn names(&self) -> Vec<&str> {
        (0..self.layout.width())
            .map(|j| column_name(&self.layout, j))
            .collect()
    }
}

impl Frames for IpcFrames {
    fn pass(&mut self, out: &mut dyn FnMut(Frame) -> bool) -> Result<(), GbdtError> {
        let names: Vec<String> = self.names().into_iter().map(String::from).collect();
        for path in &self.files {
            let shown = path.display();
            let file =
                File::open(path).map_err(|e| GbdtError::msg(format!("reading {shown}: {e}")))?;
            let mut file = BufReader::new(file);
            let meta = read_file_metadata(&mut file)
                .map_err(|e| GbdtError::msg(format!("reading {shown}: {e}")))?;
            let mut at = Vec::with_capacity(names.len());
            for name in &names {
                match meta.schema.index_of(name) {
                    Some(i) => at.push(i),
                    None => {
                        return Err(GbdtError::msg(format!("{shown} has no column `{name}`")));
                    }
                }
            }
            let mut projection = at.clone();
            projection.sort_unstable();
            // Where each column of the layout sits among the projected ones.
            let order: Vec<usize> = at
                .iter()
                .map(|i| projection.binary_search(i).expect("projected"))
                .collect();
            let maps = self.dictionary_maps();
            for batch in FileReader::new(file, meta, Some(projection), None) {
                let batch = batch.map_err(|e| GbdtError::msg(format!("reading {shown}: {e}")))?;
                let n = batch.height();
                let arrays = batch.into_arrays();
                let cols = order
                    .iter()
                    .enumerate()
                    .map(|(j, &k)| self.convert(arrays[k].as_ref(), j, &maps))
                    .collect::<Result<Vec<_>, _>>()?;
                if !out(Frame::Cols { n, cols }) {
                    return Ok(());
                }
            }
        }
        Ok(())
    }
}

type Codes = HashMap<String, u32>;

impl IpcFrames {
    fn dictionary_maps(&self) -> Vec<Option<Codes>> {
        self.categories
            .iter()
            .map(|c| {
                c.as_ref().map(|cats| {
                    cats.iter()
                        .enumerate()
                        .map(|(i, s)| (s.clone(), i as u32))
                        .collect()
                })
            })
            .collect()
    }

    /// Column `j` of the layout as the libraries read it.
    fn convert(
        &self,
        a: &dyn Array,
        j: usize,
        maps: &[Option<Codes>],
    ) -> Result<Box<dyn Array>, GbdtError> {
        let p = self.layout.names.len();
        let name = column_name(&self.layout, j);
        if j >= p {
            // The label or the weight, as Float64.
            if matches!(Num::of(a), Some(Num::F64(_))) {
                return Ok(a.to_boxed());
            }
            if let Some(num) = Num::of(a) {
                return Ok(num.to_f64().boxed());
            }
            if let Some(b) = a.as_any().downcast_ref::<BooleanArray>() {
                let v: Vec<f64> = b.values().iter().map(|x| f64::from(u8::from(x))).collect();
                return Ok(PrimitiveArray::from_vec(v)
                    .with_validity(b.validity().cloned())
                    .boxed());
            }
            return Err(not_read(name, a.dtype()));
        }
        if let Some(b) = a.as_any().downcast_ref::<BooleanArray>() {
            let v: Vec<u8> = b.values().iter().map(u8::from).collect();
            return Ok(PrimitiveArray::from_vec(v)
                .with_validity(b.validity().cloned())
                .boxed());
        }
        if let ArrowDataType::Dictionary(key, _, _) = a.dtype() {
            let (Some(codes), Some(cats), Kind::Category { .. }) =
                (&maps[j], &self.categories[j], self.layout.kinds[j])
            else {
                return Err(not_read(name, a.dtype()));
            };
            return match key {
                IntegerType::Int8 => dict::<i8>(a, name, codes, cats),
                IntegerType::Int16 => dict::<i16>(a, name, codes, cats),
                IntegerType::Int32 => dict::<i32>(a, name, codes, cats),
                IntegerType::Int64 => dict::<i64>(a, name, codes, cats),
                IntegerType::UInt8 => dict::<u8>(a, name, codes, cats),
                IntegerType::UInt16 => dict::<u16>(a, name, codes, cats),
                IntegerType::UInt32 => dict::<u32>(a, name, codes, cats),
                IntegerType::UInt64 => dict::<u64>(a, name, codes, cats),
                _ => Err(not_read(name, a.dtype())),
            };
        }
        if Num::of(a).is_some() {
            return Ok(a.to_boxed());
        }
        Err(not_read(name, a.dtype()))
    }
}

fn not_read(name: &str, dtype: &ArrowDataType) -> GbdtError {
    GbdtError::msg(format!(
        "column `{name}` is {dtype:?} in the Arrow IPC file, which is not a column the \
         libraries read"
    ))
}

/// A dictionary column's codes in the Enum's categories: its keys as they
/// are when its dictionary is the categories in order, otherwise each
/// dictionary value looked up.
fn dict<K: DictionaryKey>(
    a: &dyn Array,
    name: &str,
    codes: &Codes,
    cats: &[String],
) -> Result<Box<dyn Array>, GbdtError> {
    let d = a
        .as_any()
        .downcast_ref::<DictionaryArray<K>>()
        .expect("a dictionary array of its key type");
    let values: Vec<String> =
        dictionary_strings(d.values().as_ref()).ok_or_else(|| not_read(name, a.dtype()))?;
    if values.len() <= cats.len() && values.iter().zip(cats).all(|(v, c)| v == c) {
        return Ok(d.keys().clone().boxed());
    }
    let mut lookup = Vec::with_capacity(values.len());
    for v in &values {
        match codes.get(v) {
            Some(&c) => lookup.push(c),
            None => {
                return Err(GbdtError::msg(format!(
                    "column `{name}` holds {v:?}, which is not one of its Enum's categories"
                )));
            }
        }
    }
    let keys = d.keys();
    let mapped: Vec<u32> = keys
        .values()
        .iter()
        .zip(0..)
        .map(|(&k, i)| {
            let at: Option<usize> = k.try_into().ok();
            match at.and_then(|at| lookup.get(at)) {
                Some(&c) if keys.is_valid(i) => c,
                _ => 0,
            }
        })
        .collect();
    Ok(PrimitiveArray::from_vec(mapped)
        .with_validity(keys.validity().cloned())
        .boxed())
}

fn dictionary_strings(values: &dyn Array) -> Option<Vec<String>> {
    let any = values.as_any();
    if let Some(v) = any.downcast_ref::<Utf8ViewArray>() {
        return Some(v.values_iter().map(String::from).collect());
    }
    if let Some(v) = any.downcast_ref::<Utf8Array<i32>>() {
        return Some(v.values_iter().map(String::from).collect());
    }
    if let Some(v) = any.downcast_ref::<Utf8Array<i64>>() {
        return Some(v.values_iter().map(String::from).collect());
    }
    None
}
