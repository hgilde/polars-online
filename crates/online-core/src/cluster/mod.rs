//! Online clustering (docs/CLUSTERING.md): every model here is the mean-form
//! accumulator of [`summary`] with an assignment rule in front of it.

mod kmeans;
mod summary;

pub use kmeans::{KMeans, KMeansCfg, SeedRule};
pub use summary::{ClusterSummary, FeatureMoments, SplitMix64, dist2};
