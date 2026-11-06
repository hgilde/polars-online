//! Which readiness notices a model instance has raised ([`Notified`]) and
//! the ones not yet said ([`Notice`]), moved out of `stream.rs`, which had
//! passed the size a source file is held to (task 334).

use super::*;

/// Which readiness notices one model instance has raised
/// (docs/WARMUP-AND-CONVERGENCE.md §3), each once per stream, and the
/// messages raised since the bank last drained them.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Notified {
    /// A coefficient more ridge than data was named.
    pub support: bool,
    /// A readiness floor was found unreachable at steady state: the noise
    /// gate, or a target's `min_weight` (docs/PLAN.md task 198). One flag for
    /// both: the floor is checked first, so while it withholds a row the
    /// noise gate's reason is never the row's, and a stream says the one
    /// that holds it back.
    pub unreachable: bool,
    /// Raised and not yet drained; not state.
    #[serde(skip)]
    pub pending: Vec<Notice>,
    /// Per target, the decay time at the first row of the run of rows the
    /// `min_weight` floor has withheld since the stream was 95% settled
    /// ([`settled_enough`]); `None` where the floor let the last row
    /// through. State since schema 46: held in memory alone, a save and
    /// load started the wait again, so a stream saved and loaded each day,
    /// with a half-life longer than one run's learned clock, never said its
    /// notice (review round 5, F3; task 208). Skipped while no target
    /// waits, and emptied once a notice is said, after which nothing reads
    /// it.
    #[serde(default, skip_serializing_if = "no_wait")]
    pub floor_since: Vec<Option<f64>>,
    /// The same for the noise gate, over the instance's slots.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gate_since: Option<f64>,
    /// A target's learning rate was found past its stability bound
    /// (`unstable_rate`; task 278, D8). Written only once set, so a state
    /// that never met it keeps its bytes.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub unstable: bool,
    /// The rows the instance has learned since a reset restarted it, `None`
    /// before any: what bounds the rounding of the weight the "not met"
    /// notices project from ([`settled_weight`]), in place of the stream's
    /// accepted rows, through most of which the rebuilt model's weight was
    /// never carried (review round 10, S7-7). Schema 64; written only once
    /// set, so a state never restarted keeps its bytes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub learned_since_restart: Option<u64>,
}

/// A readiness notice raised and not yet said: its text, and the session
/// span that raised it, where one can.
///
/// Under `group_close = "session"` each span of a group is a fresh stream,
/// which says each notice once, so two spans of one group can raise the same
/// text, in one call or in two; the span's session value tells them apart
/// (task 334). It is named when the notice is taken from the span's stream
/// ([`Stream::take_notices`]), from the value the stream already holds for
/// its closed row ([`Persisted::last_session`]): at a close, before the
/// stream is replaced; otherwise once the call's rows are learned. Under any
/// other spec no session starts a stream over -- a session that restarts
/// one (`session_gap = "reset"`, a drift reset, `restart_after_step_back`)
/// rebuilds its models and keeps which notices it has said -- so a notice
/// is said once per (spec, group, instance) and names no span.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Notice {
    /// ` session "a"` (the value quoted as a group key is), or ` session
    /// null` for a span whose session value is null; `None` until the
    /// notice is taken, and for a spec that does not close on session.
    pub span: Option<String>,
    /// What the notice says, after its spec, group and span.
    pub body: String,
}

/// How a notice names the session span whose value is `value`: quoted as a
/// group key is, and a null value as `null`, which no quoted value reads.
pub(super) fn span_name(value: Option<&str>) -> String {
    value.map_or_else(
        || " session null".to_string(),
        |v| format!(" session {v:?}"),
    )
}

/// [`Notified::floor_since`] holds no wait: no target's floor is in a run.
fn no_wait(since: &[Option<f64>]) -> bool {
    since.iter().all(Option::is_none)
}

impl Notified {
    /// How long target `tj`'s floor has withheld every row: the decay time
    /// `now` less where the run began, NaN outside a run that began once the
    /// stream was `judged` settled enough ([`withheld_for`]).
    pub(super) fn floor_withheld_for(
        &mut self,
        tj: usize,
        withheld: bool,
        judged: bool,
        now: f64,
    ) -> f64 {
        if self.floor_since.len() <= tj {
            self.floor_since.resize(tj + 1, None);
        }
        withheld_for(&mut self.floor_since[tj], withheld, judged, now)
    }

    /// Hold a notice this instance has raised, until the bank takes it.
    pub(super) fn raise(&mut self, body: String) {
        self.pending.push(Notice { span: None, body });
    }

    /// A reset restarts the decay time the waits are read on, and a notice
    /// said ends them: nothing reads a wait once `unreachable` is set.
    pub(super) fn restart_waits(&mut self) {
        self.floor_since.clear();
        self.gate_since = None;
    }
}
