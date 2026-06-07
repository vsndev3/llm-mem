use std::sync::Arc;

use tokio::sync::Semaphore;

use crate::llm::LLMClient;

// ── Priority ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlmPriority {
    /// User-facing queries: search, classification. High concurrency.
    Interactive,
    /// Background pipeline: ingestion, abstraction, auto-enhance. Low concurrency.
    Background,
}

// ── PriorityLLMClient ───────────────────────────────────────────────────────

/// Wraps an `LLMClient` with priority-aware concurrency control.
///
/// **Interactive** calls (search, query intent classification) get a high permit
/// limit so user queries are never starved. **Background** calls (ingestion,
/// abstraction pipeline) get a low permit limit to prevent them from saturating
/// the upstream LLM API.
///
/// # Usage
///
/// ```ignore
/// let _permit = priority_client.acquire(LlmPriority::Interactive).await;
/// priority_client.inner().complete("hello").await?;
/// // _permit dropped here — permit returned to pool
/// ```
pub struct PriorityLLMClient {
    inner: Box<dyn LLMClient>,
    interactive_sem: Arc<Semaphore>,
    background_sem: Arc<Semaphore>,
}

impl PriorityLLMClient {
    /// Create a new priority-aware LLM client.
    ///
    /// `interactive_permits` — max concurrent interactive calls (default: 10).
    /// `background_permits` — max concurrent background calls (default: 3).
    pub fn new(
        inner: Box<dyn LLMClient>,
        interactive_permits: usize,
        background_permits: usize,
    ) -> Self {
        Self {
            inner,
            interactive_sem: Arc::new(Semaphore::new(interactive_permits)),
            background_sem: Arc::new(Semaphore::new(background_permits)),
        }
    }

    /// Acquire a priority permit. The returned `OwnedSemaphorePermit` is
    /// automatically released on drop, returning the permit to the pool.
    ///
    /// Callers wrap their LLM invocation with this guard to limit concurrency:
    ///
    /// ```ignore
    /// let _guard = client.acquire(LlmPriority::Interactive).await;
    /// let result = client.inner().complete(prompt).await;
    /// ```
    pub async fn acquire(&self, priority: LlmPriority) -> tokio::sync::OwnedSemaphorePermit {
        match priority {
            LlmPriority::Interactive => self
                .interactive_sem
                .clone()
                .acquire_owned()
                .await
                .expect("interactive semaphore closed"),
            LlmPriority::Background => self
                .background_sem
                .clone()
                .acquire_owned()
                .await
                .expect("background semaphore closed"),
        }
    }

    /// Get a reference to the inner LLM client for making calls after acquiring a permit.
    pub fn inner(&self) -> &dyn LLMClient {
        self.inner.as_ref()
    }

    /// Get the current number of available interactive permits.
    pub fn available_interactive(&self) -> usize {
        self.interactive_sem.available_permits()
    }

    /// Get the current number of available background permits.
    pub fn available_background(&self) -> usize {
        self.background_sem.available_permits()
    }
}
