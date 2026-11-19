//! Every problem a check of a model's settings finds, reported at once.
//!
//! A check that returned at its first problem made a reader fix one
//! setting, run again and meet the next (docs/WRITING.md §11; the user's
//! decision 2 of task 338). A [`Refusals`] collects each problem as a whole
//! message -- where it happened, what is wrong with the value, and the
//! remedy as a sentence of its own -- and the check returns them together,
//! one to a line. A caller that adds its own problems, as the spec layer
//! does, reads the lines back with [`Refusals::take`] and renders the whole
//! set once.
//!
//! Where one problem makes the checks after it meaningless, such as a
//! dimension of 0 before the checks that index by it, the check stops there
//! with [`Refusals::stop_here`], and the refusal ends with [`STOPPED`] so
//! the reader knows more may follow.

/// The last line of a refusal whose check stopped before it had read every
/// setting.
pub const STOPPED: &str = "Fix this, then run again to check the other settings.";
/// [`STOPPED`] after several problems.
pub const STOPPED_MANY: &str = "Fix these, then run again to check the other settings.";

/// The problems one check has found so far, in the order it found them.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct Refusals {
    found: Vec<String>,
    stopped: bool,
}

impl Refusals {
    pub fn new() -> Self {
        Self::default()
    }

    /// One problem, written as a whole message.
    pub fn refuse(&mut self, problem: impl Into<String>) {
        let problem = problem.into();
        // A line of its own each: a problem is one line of the refusal.
        self.found.extend(
            problem
                .lines()
                .filter(|l| !l.trim().is_empty())
                .map(str::to_string),
        );
    }

    /// Whether no problem has been found.
    pub fn is_empty(&self) -> bool {
        self.found.is_empty()
    }

    /// The problems found so far.
    pub fn problems(&self) -> &[String] {
        &self.found
    }

    /// Whether a check stopped before it had read every setting.
    pub fn stopped(&self) -> bool {
        self.stopped
    }

    /// Another check's result: `true` when it found nothing, and otherwise
    /// each problem its refusal lists, added to these, and its stop with
    /// them.
    pub fn take(&mut self, result: Result<(), String>) -> bool {
        self.value(result).is_some()
    }

    /// [`Self::take`] for a check that returns a value when it finds
    /// nothing.
    pub fn value<T>(&mut self, result: Result<T, String>) -> Option<T> {
        match result {
            Ok(v) => Some(v),
            Err(e) => {
                let (problems, stopped) = Self::parse(&e);
                self.found.extend(problems.into_iter().map(str::to_string));
                self.stopped |= stopped;
                None
            }
        }
    }

    /// A point the check cannot go past with a problem found: the checks
    /// after it read what the problems refuse. With none found, `Ok`.
    pub fn stop_here(&mut self) -> Result<(), String> {
        if self.found.is_empty() {
            return Ok(());
        }
        self.stopped = true;
        Err(self.render())
    }

    /// The refusal, if any problem was found.
    pub fn finish(self) -> Result<(), String> {
        if self.found.is_empty() {
            Ok(())
        } else {
            Err(self.render())
        }
    }

    /// The problems, one to a line, and [`STOPPED`] last where the check
    /// stopped.
    fn render(&self) -> String {
        let mut lines = self.found.clone();
        if self.stopped {
            let last = if self.found.len() == 1 {
                STOPPED
            } else {
                STOPPED_MANY
            };
            lines.push(last.to_string());
        }
        lines.join("\n")
    }

    /// The problems a refusal lists, and whether its check stopped: the
    /// lines of a refusal [`Refusals`] rendered, or one problem for any
    /// other message.
    pub fn parse(refusal: &str) -> (Vec<&str>, bool) {
        let mut stopped = false;
        let problems = refusal
            .lines()
            .filter(|l| {
                let l = l.trim();
                let last = l == STOPPED || l == STOPPED_MANY;
                stopped |= last;
                !l.is_empty() && !last
            })
            .collect();
        (problems, stopped)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_problem_is_kept_in_order_and_a_stop_is_said() {
        let mut r = Refusals::new();
        assert!(r.is_empty() && r.stop_here().is_ok());
        r.refuse("m: a is wrong. Set a.");
        assert!(!r.is_empty());
        assert!(!r.take(Err("m: b is wrong. Set b.\nm: c is wrong. Set c.".into())));
        assert!(r.take(Ok(())));
        assert_eq!(r.problems().len(), 3);
        assert!(!r.stopped());
        assert_eq!(r.clone().finish().unwrap_err().lines().count(), 3);
        let e = r.stop_here().unwrap_err();
        assert_eq!(
            e,
            format!(
                "m: a is wrong. Set a.\nm: b is wrong. Set b.\nm: c is wrong. Set c.\n{STOPPED_MANY}"
            )
        );
        // A stop travels with the problems it ended.
        let mut outer = Refusals::new();
        assert_eq!(outer.value::<u8>(Err(e)), None);
        assert!(outer.stopped() && outer.problems().len() == 3);
        assert_eq!(Refusals::parse("one line"), (vec!["one line"], false));
        assert!(Refusals::new().finish().is_ok());
    }
}
