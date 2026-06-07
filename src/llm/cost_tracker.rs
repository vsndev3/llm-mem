use std::sync::atomic::{AtomicU64, Ordering};

/// Tracks LLM token consumption against a per-session budget.
/// Thread-safe via atomics — shared across all LLM operations in a session.
#[derive(Debug)]
pub struct LlmCostTracker {
    tokens_used: AtomicU64,
    budget: u64,
    dry_run: bool,
}

impl LlmCostTracker {
    pub fn new(budget: u64, dry_run: bool) -> Self {
        Self {
            tokens_used: AtomicU64::new(0),
            budget,
            dry_run,
        }
    }

    pub fn check_budget(&self, estimated_tokens: u64) -> Result<u64, crate::error::MemoryError> {
        if self.dry_run {
            return Ok(0);
        }
        if self.budget == 0 {
            return Ok(0);
        }
        let used = self.tokens_used.load(Ordering::Relaxed);
        let new_total = used.saturating_add(estimated_tokens);
        if new_total > self.budget {
            return Err(crate::error::MemoryError::LLM(format!(
                "Token budget exceeded: {} used + {} estimated = {} > {} budget",
                used, estimated_tokens, new_total, self.budget
            )));
        }
        Ok(self.budget.saturating_sub(new_total))
    }

    pub fn record(&self, tokens: u64) {
        self.tokens_used.fetch_add(tokens, Ordering::Relaxed);
    }

    pub fn used(&self) -> u64 {
        self.tokens_used.load(Ordering::Relaxed)
    }

    pub fn remaining(&self) -> Option<u64> {
        if self.budget == 0 {
            None
        } else {
            Some(self.budget.saturating_sub(self.used()))
        }
    }

    pub fn is_dry_run(&self) -> bool {
        self.dry_run
    }
}

impl Default for LlmCostTracker {
    fn default() -> Self {
        Self::new(0, false)
    }
}
