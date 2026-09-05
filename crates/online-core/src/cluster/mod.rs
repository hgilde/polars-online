//! Online clustering (docs/CLUSTERING.md): every model here is the mean-form
//! accumulator of [`summary`] with an assignment rule in front of it.

mod kmeans;
mod micro;
mod summary;

pub use kmeans::{KMeans, KMeansCfg, SeedRule};
pub use micro::{LINK_FACTOR, LINK_FLOOR, LINK_QUANTILE, Micro, MicroCfg, MicroCluster};
pub use summary::{ClusterSummary, FeatureMoments, SplitMix64, dist2, merged_radius2};
