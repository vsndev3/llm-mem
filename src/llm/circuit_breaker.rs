use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use tracing::{debug, warn};

use crate::error::MemoryError;
use crate::llm::client::LLMClient;

// ── Circuit Breaker State Machine ──────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    Closed,
    Open,
    HalfOpen,
}

impl std::fmt::Display for CircuitState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CircuitState::Closed => write!(f, "Closed"),
            CircuitState::Open => write!(f, "Open"),
            CircuitState::HalfOpen => write!(f, "HalfOpen"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    pub failure_threshold: usize,
    pub success_threshold: usize,
    pub cooldown: Duration,
    pub half_open_max: usize,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            success_threshold: 2,
            cooldown: Duration::from_secs(30),
            half_open_max: 1,
        }
    }
}

struct CircuitBreakerInner {
    state: CircuitState,
    opened_at: Option<Instant>,
    consecutive_failures: usize,
    consecutive_successes: usize,
    half_open_in_flight: usize,
}

impl Default for CircuitBreakerInner {
    fn default() -> Self {
        Self {
            state: CircuitState::Closed,
            opened_at: None,
            consecutive_failures: 0,
            consecutive_successes: 0,
            half_open_in_flight: 0,
        }
    }
}

/// Circuit breaker that wraps LLM calls to fail fast during outages.
///
/// Three states:
/// - **Closed**: Normal operation. Failures are counted.
/// - **Open**: Fail fast after `failure_threshold` consecutive failures. After `cooldown` transitions to HalfOpen.
/// - **HalfOpen**: Allow `half_open_max` probe requests through. Successes transition to Closed, failures back to Open.
#[derive(Clone)]
pub struct CircuitBreaker {
    inner: Arc<RwLock<CircuitBreakerInner>>,
    config: CircuitBreakerConfig,
}

impl Default for CircuitBreaker {
    fn default() -> Self {
        Self::new(CircuitBreakerConfig::default())
    }
}

impl CircuitBreaker {
    pub fn new(config: CircuitBreakerConfig) -> Self {
        Self {
            inner: Arc::new(RwLock::new(CircuitBreakerInner::default())),
            config,
        }
    }

    pub fn with_config(
        failure_threshold: usize,
        success_threshold: usize,
        cooldown: Duration,
    ) -> Self {
        Self::new(CircuitBreakerConfig {
            failure_threshold,
            success_threshold,
            cooldown,
            half_open_max: 1,
        })
    }

    /// Check if the Open state has expired and should transition to HalfOpen.
    /// This is called before any state-modifying operation to ensure consistency.
    fn maybe_transition_to_half_open(inner: &mut CircuitBreakerInner, cooldown: Duration) {
        if let (CircuitState::Open, Some(opened_at)) = (inner.state, inner.opened_at)
            && opened_at.elapsed() >= cooldown
        {
            inner.state = CircuitState::HalfOpen;
            inner.half_open_in_flight = 0;
            inner.consecutive_successes = 0;
            debug!("Circuit breaker transitioning Open -> HalfOpen");
        }
    }

    /// Check current state and potentially allow a request through.
    /// Returns `true` if the request should proceed.
    pub fn allow_request(&self) -> bool {
        let mut inner = self.inner.write().unwrap();
        Self::maybe_transition_to_half_open(&mut inner, self.config.cooldown);

        match inner.state {
            CircuitState::Closed => true,
            CircuitState::HalfOpen => {
                if inner.half_open_in_flight < self.config.half_open_max {
                    inner.half_open_in_flight += 1;
                    true
                } else {
                    false
                }
            }
            CircuitState::Open => false,
        }
    }

    /// Record a successful call.
    pub fn record_success(&self) {
        let mut inner = self.inner.write().unwrap();
        Self::maybe_transition_to_half_open(&mut inner, self.config.cooldown);

        match inner.state {
            CircuitState::HalfOpen => {
                inner.half_open_in_flight = inner.half_open_in_flight.saturating_sub(1);
                inner.consecutive_successes += 1;

                if inner.consecutive_successes >= self.config.success_threshold {
                    inner.state = CircuitState::Closed;
                    inner.consecutive_failures = 0;
                    debug!("Circuit breaker transitioning HalfOpen -> Closed");
                }
            }
            CircuitState::Closed => {
                inner.consecutive_failures = 0;
                inner.consecutive_successes += 1;
            }
            CircuitState::Open => {
                // Shouldn't happen since we transition above, but handle gracefully
            }
        }
    }

    /// Record a failed call.
    pub fn record_failure(&self) {
        let mut inner = self.inner.write().unwrap();
        Self::maybe_transition_to_half_open(&mut inner, self.config.cooldown);

        match inner.state {
            CircuitState::HalfOpen => {
                inner.half_open_in_flight = inner.half_open_in_flight.saturating_sub(1);
                inner.state = CircuitState::Open;
                inner.opened_at = Some(Instant::now());
                inner.consecutive_failures = 1;
                inner.consecutive_successes = 0;
                warn!("Circuit breaker transitioning HalfOpen -> Open (probe failed)");
            }
            CircuitState::Closed => {
                inner.consecutive_failures += 1;
                if inner.consecutive_failures >= self.config.failure_threshold {
                    inner.state = CircuitState::Open;
                    inner.opened_at = Some(Instant::now());
                    inner.consecutive_successes = 0;
                    warn!(
                        "Circuit breaker transitioning Closed -> Open ({} consecutive failures)",
                        inner.consecutive_failures
                    );
                }
            }
            CircuitState::Open => {
                // Already open, no-op
            }
        }
    }

    /// Get the current state for inspection.
    pub fn get_state(&self) -> CircuitState {
        let mut inner = self.inner.write().unwrap();
        Self::maybe_transition_to_half_open(&mut inner, self.config.cooldown);
        inner.state
    }

    /// Get remaining cooldown time when circuit is Open.
    pub fn remaining_cooldown(&self) -> Option<Duration> {
        let mut inner = self.inner.write().unwrap();
        if let (CircuitState::Open, Some(opened_at)) = (inner.state, inner.opened_at) {
            let elapsed = opened_at.elapsed();
            if elapsed < self.config.cooldown {
                return Some(self.config.cooldown - elapsed);
            }
            Self::maybe_transition_to_half_open(&mut inner, self.config.cooldown);
        }
        None
    }

    /// Get stats for monitoring.
    pub fn stats(&self) -> CircuitBreakerStats {
        let inner = self.inner.read().unwrap();
        CircuitBreakerStats {
            state: inner.state,
            consecutive_failures: inner.consecutive_failures,
            consecutive_successes: inner.consecutive_successes,
            half_open_in_flight: inner.half_open_in_flight,
        }
    }
}

/// Stats snapshot for monitoring.
#[derive(Debug, Clone)]
pub struct CircuitBreakerStats {
    pub state: CircuitState,
    pub consecutive_failures: usize,
    pub consecutive_successes: usize,
    pub half_open_in_flight: usize,
}

// ── Exponential Backoff with Jitter ────────────────────────────────────────

/// Calculate backoff duration with exponential growth and random jitter.
///
/// Formula: `min(base * 2^attempt, max) + random(0, jitter)`
pub fn backoff_duration(
    attempt: u32,
    base_ms: u64,
    max_ms: u64,
    jitter_ms: u64,
) -> Duration {
    let exp = base_ms.saturating_mul(2u64.saturating_pow(attempt));
    let capped = exp.min(max_ms);
    let jitter = if jitter_ms > 0 {
        rand::random::<u64>() % (jitter_ms + 1)
    } else {
        0
    };
    Duration::from_millis(capped + jitter)
}

// ── Circuit Breaker LLM Client Wrapper ─────────────────────────────────────

/// Wrapper that applies circuit breaker and exponential backoff to all LLM calls.
#[derive(Clone)]
pub struct CircuitBreakerLLMClient {
    inner: Box<dyn LLMClient>,
    circuit_breaker: CircuitBreaker,
    max_retries: u32,
    base_backoff_ms: u64,
    max_backoff_ms: u64,
    jitter_ms: u64,
}

impl CircuitBreakerLLMClient {
    pub fn new(
        inner: Box<dyn LLMClient>,
        circuit_breaker: CircuitBreaker,
        max_retries: u32,
        base_backoff_ms: u64,
        max_backoff_ms: u64,
        jitter_ms: u64,
    ) -> Self {
        Self {
            inner,
            circuit_breaker,
            max_retries,
            base_backoff_ms,
            max_backoff_ms,
            jitter_ms,
        }
    }

    pub fn with_defaults(inner: Box<dyn LLMClient>) -> Self {
        Self {
            inner,
            circuit_breaker: CircuitBreaker::with_config(5, 2, Duration::from_secs(30)),
            max_retries: 3,
            base_backoff_ms: 100,
            max_backoff_ms: 10_000,
            jitter_ms: 500,
        }
    }

    pub fn get_circuit_breaker(&self) -> CircuitBreaker {
        self.circuit_breaker.clone()
    }

    async fn call_with_retry<F, Fut, T>(&self, mut f: F) -> crate::error::Result<T>
    where
        F: FnMut() -> Fut,
        Fut: std::future::Future<Output = crate::error::Result<T>>,
    {
        let mut last_err: Option<String> = None;

        for attempt in 0..self.max_retries {
            if !self.circuit_breaker.allow_request() {
                let remaining = self
                    .circuit_breaker
                    .remaining_cooldown()
                    .map(|d| format!("{}ms remaining", d.as_millis()))
                    .unwrap_or_else(|| "cooldown".to_string());
                return Err(MemoryError::LLM(format!(
                    "Circuit breaker Open: {}",
                    remaining
                )));
            }

            match f().await {
                Ok(v) => {
                    self.circuit_breaker.record_success();
                    return Ok(v);
                }
                Err(e) => {
                    self.circuit_breaker.record_failure();
                    last_err = Some(e.to_string());

                    if attempt < self.max_retries - 1 {
                        let delay = backoff_duration(
                            attempt,
                            self.base_backoff_ms,
                            self.max_backoff_ms,
                            self.jitter_ms,
                        );
                        debug!(
                            "Retry attempt {} failed: {}. Backing off {:?} before next attempt.",
                            attempt + 1,
                            e,
                            delay
                        );
                        tokio::time::sleep(delay).await;
                    }
                }
            }
        }

        Err(MemoryError::LLM(format!(
            "All {} retry attempts exhausted: {}",
            self.max_retries,
            last_err.unwrap_or("unknown error".into())
        )))
    }
}

#[async_trait]
impl LLMClient for CircuitBreakerLLMClient {
    async fn complete(&self, prompt: &str) -> crate::error::Result<String> {
        let prompt = prompt.to_string();
        self.call_with_retry(|| async { self.inner.complete(&prompt).await })
            .await
    }

    async fn complete_with_grammar(&self, prompt: &str, grammar: &str) -> crate::error::Result<String> {
        let prompt = prompt.to_string();
        let grammar = grammar.to_string();
        self.call_with_retry(|| async { self.inner.complete_with_grammar(&prompt, &grammar).await })
            .await
    }

    async fn embed(&self, text: &str) -> crate::error::Result<Vec<f32>> {
        let text = text.to_string();
        self.call_with_retry(|| async { self.inner.embed(&text).await }).await
    }

    async fn embed_batch(&self, texts: &[String]) -> crate::error::Result<Vec<Vec<f32>>> {
        let texts = texts.to_vec();
        self.call_with_retry(|| async { self.inner.embed_batch(&texts).await })
            .await
    }

    async fn extract_keywords(&self, content: &str) -> crate::error::Result<Vec<String>> {
        let content = content.to_string();
        self.call_with_retry(|| async { self.inner.extract_keywords(&content).await })
            .await
    }

    async fn summarize(&self, content: &str, max_length: Option<usize>) -> crate::error::Result<String> {
        let content = content.to_string();
        self.call_with_retry(|| async { self.inner.summarize(&content, max_length).await })
            .await
    }

    async fn health_check(&self) -> crate::error::Result<bool> {
        self.call_with_retry(|| async { self.inner.health_check().await })
            .await
    }

    async fn extract_structured_facts(
        &self,
        prompt: &str,
    ) -> crate::error::Result<crate::llm::StructuredFactExtraction> {
        let prompt = prompt.to_string();
        self.call_with_retry(|| async { self.inner.extract_structured_facts(&prompt).await })
            .await
    }

    async fn extract_detailed_facts(
        &self,
        prompt: &str,
    ) -> crate::error::Result<crate::llm::DetailedFactExtraction> {
        let prompt = prompt.to_string();
        self.call_with_retry(|| async { self.inner.extract_detailed_facts(&prompt).await })
            .await
    }

    async fn extract_keywords_structured(
        &self,
        prompt: &str,
    ) -> crate::error::Result<crate::llm::KeywordExtraction> {
        let prompt = prompt.to_string();
        self.call_with_retry(|| async { self.inner.extract_keywords_structured(&prompt).await })
            .await
    }

    async fn classify_memory(&self, prompt: &str) -> crate::error::Result<crate::llm::MemoryClassification> {
        let prompt = prompt.to_string();
        self.call_with_retry(|| async { self.inner.classify_memory(&prompt).await })
            .await
    }

    async fn score_importance(&self, prompt: &str) -> crate::error::Result<crate::llm::ImportanceScore> {
        let prompt = prompt.to_string();
        self.call_with_retry(|| async { self.inner.score_importance(&prompt).await })
            .await
    }

    async fn check_duplicates(&self, prompt: &str) -> crate::error::Result<crate::llm::DeduplicationResult> {
        let prompt = prompt.to_string();
        self.call_with_retry(|| async { self.inner.check_duplicates(&prompt).await })
            .await
    }

    async fn generate_summary(&self, prompt: &str) -> crate::error::Result<crate::llm::SummaryResult> {
        let prompt = prompt.to_string();
        self.call_with_retry(|| async { self.inner.generate_summary(&prompt).await })
            .await
    }

    async fn detect_language(&self, prompt: &str) -> crate::error::Result<crate::llm::LanguageDetection> {
        let prompt = prompt.to_string();
        self.call_with_retry(|| async { self.inner.detect_language(&prompt).await })
            .await
    }

    async fn extract_entities(&self, prompt: &str) -> crate::error::Result<crate::llm::EntityExtraction> {
        let prompt = prompt.to_string();
        self.call_with_retry(|| async { self.inner.extract_entities(&prompt).await })
            .await
    }

    async fn analyze_conversation(
        &self,
        prompt: &str,
    ) -> crate::error::Result<crate::llm::ConversationAnalysis> {
        let prompt = prompt.to_string();
        self.call_with_retry(|| async { self.inner.analyze_conversation(&prompt).await })
            .await
    }

    async fn extract_metadata_enrichment(
        &self,
        prompt: &str,
    ) -> crate::error::Result<crate::llm::MetadataEnrichment> {
        let prompt = prompt.to_string();
        self.call_with_retry(|| async { self.inner.extract_metadata_enrichment(&prompt).await })
            .await
    }

    async fn extract_metadata_enrichment_batch(
        &self,
        texts: &[String],
    ) -> crate::error::Result<Vec<crate::error::Result<crate::llm::MetadataEnrichment>>> {
        let texts = texts.to_vec();
        self.call_with_retry(|| async { self.inner.extract_metadata_enrichment_batch(&texts).await })
            .await
    }

    async fn complete_batch(&self, prompts: &[String]) -> crate::error::Result<Vec<crate::error::Result<String>>> {
        let prompts = prompts.to_vec();
        self.call_with_retry(|| async { self.inner.complete_batch(&prompts).await })
            .await
    }

    fn get_status(&self) -> crate::llm::ClientStatus {
        let mut status = self.inner.get_status();
        let stats = self.circuit_breaker.stats();
        status.details.insert(
            "circuit_breaker_state".into(),
            serde_json::Value::String(stats.state.to_string()),
        );
        status.details.insert(
            "circuit_breaker_consecutive_failures".into(),
            serde_json::Value::Number(serde_json::Number::from(stats.consecutive_failures)),
        );
        status.details.insert(
            "circuit_breaker_consecutive_successes".into(),
            serde_json::Value::Number(serde_json::Number::from(stats.consecutive_successes)),
        );
        status
    }

    fn batch_config(&self) -> (usize, u32) {
        self.inner.batch_config()
    }

    async fn enhance_memory_unified(&self, prompt: &str) -> crate::error::Result<crate::llm::MemoryEnhancement> {
        let prompt = prompt.to_string();
        self.call_with_retry(|| async { self.inner.enhance_memory_unified(&prompt).await })
            .await
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    // ── Backoff tests ──

    #[test]
    fn test_backoff_attempt_zero() {
        let d = backoff_duration(0, 100, 10_000, 0);
        assert_eq!(d, Duration::from_millis(100));
    }

    #[test]
    fn test_backoff_attempt_one() {
        let d = backoff_duration(1, 100, 10_000, 0);
        assert_eq!(d, Duration::from_millis(200));
    }

    #[test]
    fn test_backoff_attempt_two() {
        let d = backoff_duration(2, 100, 10_000, 0);
        assert_eq!(d, Duration::from_millis(400));
    }

    #[test]
    fn test_backoff_attempt_high() {
        let d = backoff_duration(5, 100, 10_000, 0);
        assert_eq!(d, Duration::from_millis(3200));
    }

    #[test]
    fn test_backoff_caps_at_max() {
        let d = backoff_duration(20, 100, 500, 0);
        assert_eq!(d, Duration::from_millis(500));
    }

    #[test]
    fn test_backoff_with_jitter_range() {
        for _ in 0..100 {
            let d = backoff_duration(0, 100, 10_000, 50);
            assert!(d >= Duration::from_millis(100));
            assert!(d <= Duration::from_millis(150));
        }
    }

    #[test]
    fn test_backoff_zero_jitter() {
        let d = backoff_duration(3, 100, 10_000, 0);
        assert_eq!(d, Duration::from_millis(800));
    }

    #[test]
    fn test_backoff_saturating_pow_overflow() {
        let d = backoff_duration(100, 1000, 5000, 0);
        assert_eq!(d, Duration::from_millis(5000));
    }

    // ── Circuit breaker state transitions ──

    #[test]
    fn test_circuit_breaker_starts_closed() {
        let cb = CircuitBreaker::default();
        assert_eq!(cb.get_state(), CircuitState::Closed);
    }

    #[test]
    fn test_circuit_breaker_opens_after_threshold() {
        let cb = CircuitBreaker::with_config(3, 2, Duration::from_millis(100));

        for _ in 0..2 {
            cb.record_failure();
            assert_eq!(cb.get_state(), CircuitState::Closed);
        }

        cb.record_failure();
        assert_eq!(cb.get_state(), CircuitState::Open);
    }

    #[test]
    fn test_circuit_breaker_resets_on_success() {
        let cb = CircuitBreaker::with_config(3, 2, Duration::from_millis(100));

        cb.record_failure();
        cb.record_failure();
        assert_eq!(cb.get_state(), CircuitState::Closed);

        cb.record_success();
        cb.record_failure();
        assert_eq!(cb.get_state(), CircuitState::Closed);
    }

    #[tokio::test]
    async fn test_circuit_breaker_transitions_to_half_open() {
        let cb = CircuitBreaker::with_config(2, 2, Duration::from_millis(50));

        cb.record_failure();
        cb.record_failure();
        assert_eq!(cb.get_state(), CircuitState::Open);

        tokio::time::sleep(Duration::from_millis(60)).await;
        assert_eq!(cb.get_state(), CircuitState::HalfOpen);
    }

    #[tokio::test]
    async fn test_circuit_breaker_half_open_to_closed() {
        let cb = CircuitBreaker::with_config(2, 2, Duration::from_millis(50));

        cb.record_failure();
        cb.record_failure();
        tokio::time::sleep(Duration::from_millis(60)).await;

        assert_eq!(cb.get_state(), CircuitState::HalfOpen);

        cb.record_success();
        assert_eq!(cb.get_state(), CircuitState::HalfOpen);

        cb.record_success();
        assert_eq!(cb.get_state(), CircuitState::Closed);
    }

    #[tokio::test]
    async fn test_circuit_breaker_half_open_failure_back_to_open() {
        let cb = CircuitBreaker::with_config(2, 2, Duration::from_millis(50));

        cb.record_failure();
        cb.record_failure();
        tokio::time::sleep(Duration::from_millis(60)).await;

        assert_eq!(cb.get_state(), CircuitState::HalfOpen);
        cb.record_failure();
        assert_eq!(cb.get_state(), CircuitState::Open);
    }

    #[test]
    fn test_circuit_breaker_allow_request_closed() {
        let cb = CircuitBreaker::default();
        assert!(cb.allow_request());
    }

    #[test]
    fn test_circuit_breaker_rejects_when_open() {
        let cb = CircuitBreaker::with_config(2, 2, Duration::from_secs(10));

        cb.record_failure();
        cb.record_failure();

        assert!(!cb.allow_request());
    }

    #[tokio::test]
    async fn test_circuit_breaker_allows_after_cooldown() {
        let cb = CircuitBreaker::with_config(2, 2, Duration::from_millis(50));

        cb.record_failure();
        cb.record_failure();

        tokio::time::sleep(Duration::from_millis(60)).await;
        assert!(cb.allow_request());
    }

    #[tokio::test]
    async fn test_circuit_breaker_half_open_probe_limit() {
        let config = CircuitBreakerConfig {
            failure_threshold: 2,
            success_threshold: 2,
            cooldown: Duration::from_millis(50),
            half_open_max: 1,
        };
        let cb = CircuitBreaker::new(config);

        cb.record_failure();
        cb.record_failure();
        tokio::time::sleep(Duration::from_millis(60)).await;

        assert!(cb.allow_request());
        assert!(!cb.allow_request());
    }

    #[test]
    fn test_circuit_breaker_stats() {
        let cb = CircuitBreaker::with_config(3, 2, Duration::from_secs(10));

        cb.record_failure();
        cb.record_failure();

        let stats = cb.stats();
        assert_eq!(stats.consecutive_failures, 2);
        assert_eq!(stats.consecutive_successes, 0);
        assert_eq!(stats.state, CircuitState::Closed);
    }

    #[tokio::test]
    async fn test_circuit_breaker_remaining_cooldown() {
        let cb = CircuitBreaker::with_config(2, 2, Duration::from_millis(200));

        cb.record_failure();
        cb.record_failure();

        let remaining = cb.remaining_cooldown();
        assert!(remaining.is_some());
        assert!(remaining.unwrap() > Duration::from_millis(0));

        tokio::time::sleep(Duration::from_millis(250)).await;
        assert!(cb.remaining_cooldown().is_none());
    }

    #[test]
    fn test_circuit_breaker_state_display() {
        assert_eq!(CircuitState::Closed.to_string(), "Closed");
        assert_eq!(CircuitState::Open.to_string(), "Open");
        assert_eq!(CircuitState::HalfOpen.to_string(), "HalfOpen");
    }

    // ── Mock client for wrapper tests ──

    #[derive(Clone)]
    struct MockClient {
        call_count: Arc<AtomicU32>,
        fail_count: Arc<AtomicU32>,
        should_fail: Arc<AtomicU32>,
    }

    impl MockClient {
        fn new() -> Self {
            Self {
                call_count: Arc::new(AtomicU32::new(0)),
                fail_count: Arc::new(AtomicU32::new(0)),
                should_fail: Arc::new(AtomicU32::new(0)),
            }
        }
    }

    #[async_trait]
    impl LLMClient for MockClient {
        async fn complete(&self, prompt: &str) -> crate::error::Result<String> {
            self.call_count.fetch_add(1, Ordering::Relaxed);
            let fails = self.should_fail.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| {
                if v > 0 { Some(v - 1) } else { Some(0) }
            }).unwrap_or(0);
            if fails > 0 {
                self.fail_count.fetch_add(1, Ordering::Relaxed);
                return Err(MemoryError::LLM("mock failure".into()));
            }
            Ok(format!("mock: {}", prompt))
        }

        async fn complete_with_grammar(&self, _prompt: &str, _grammar: &str) -> crate::error::Result<String> {
            Ok("{}".to_string())
        }

        async fn embed(&self, _text: &str) -> crate::error::Result<Vec<f32>> {
            Ok(vec![1.0])
        }

        async fn embed_batch(&self, texts: &[String]) -> crate::error::Result<Vec<Vec<f32>>> {
            Ok(texts.iter().map(|_| vec![1.0]).collect())
        }

        async fn extract_keywords(&self, _content: &str) -> crate::error::Result<Vec<String>> {
            Ok(vec![])
        }

        async fn summarize(&self, _content: &str, _max_length: Option<usize>) -> crate::error::Result<String> {
            Ok("".into())
        }

        async fn health_check(&self) -> crate::error::Result<bool> {
            Ok(true)
        }

        async fn extract_structured_facts(
            &self,
            _prompt: &str,
        ) -> crate::error::Result<crate::llm::StructuredFactExtraction> {
            Ok(crate::llm::StructuredFactExtraction { facts: vec![] })
        }

        async fn extract_detailed_facts(
            &self,
            _prompt: &str,
        ) -> crate::error::Result<crate::llm::DetailedFactExtraction> {
            Ok(crate::llm::DetailedFactExtraction { facts: vec![] })
        }

        async fn extract_keywords_structured(
            &self,
            _prompt: &str,
        ) -> crate::error::Result<crate::llm::KeywordExtraction> {
            Ok(crate::llm::KeywordExtraction { keywords: vec![] })
        }

        async fn classify_memory(&self, _prompt: &str) -> crate::error::Result<crate::llm::MemoryClassification> {
            Ok(crate::llm::MemoryClassification {
                memory_type: "Factual".into(),
                confidence: 1.0,
                reasoning: "".into(),
            })
        }

        async fn score_importance(&self, _prompt: &str) -> crate::error::Result<crate::llm::ImportanceScore> {
            Ok(crate::llm::ImportanceScore {
                score: 1.0,
                reasoning: "".into(),
            })
        }

        async fn check_duplicates(&self, _prompt: &str) -> crate::error::Result<crate::llm::DeduplicationResult> {
            Ok(crate::llm::DeduplicationResult {
                is_duplicate: false,
                similarity_score: 0.0,
                original_memory_id: None,
            })
        }

        async fn generate_summary(&self, _prompt: &str) -> crate::error::Result<crate::llm::SummaryResult> {
            Ok(crate::llm::SummaryResult {
                summary: "".into(),
                key_points: vec![],
            })
        }

        async fn detect_language(&self, _prompt: &str) -> crate::error::Result<crate::llm::LanguageDetection> {
            Ok(crate::llm::LanguageDetection {
                language: "en".into(),
                confidence: 1.0,
            })
        }

        async fn extract_entities(&self, _prompt: &str) -> crate::error::Result<crate::llm::EntityExtraction> {
            Ok(crate::llm::EntityExtraction { entities: vec![] })
        }

        async fn analyze_conversation(
            &self,
            _prompt: &str,
        ) -> crate::error::Result<crate::llm::ConversationAnalysis> {
            Ok(crate::llm::ConversationAnalysis {
                topics: vec![],
                sentiment: "neutral".into(),
                user_intent: "unknown".into(),
                key_information: vec![],
            })
        }

        async fn extract_metadata_enrichment(
            &self,
            _prompt: &str,
        ) -> crate::error::Result<crate::llm::MetadataEnrichment> {
            Ok(crate::llm::MetadataEnrichment {
                summary: "".into(),
                keywords: vec![],
            })
        }

        async fn extract_metadata_enrichment_batch(
            &self,
            texts: &[String],
        ) -> crate::error::Result<Vec<crate::error::Result<crate::llm::MetadataEnrichment>>> {
            Ok(texts
                .iter()
                .map(|_| {
                    Ok(crate::llm::MetadataEnrichment {
                        summary: "".into(),
                        keywords: vec![],
                    })
                })
                .collect())
        }

        async fn complete_batch(&self, prompts: &[String]) -> crate::error::Result<Vec<crate::error::Result<String>>> {
            Ok(prompts.iter().map(|_| Ok("".into())).collect())
        }

        fn get_status(&self) -> crate::llm::ClientStatus {
            crate::llm::ClientStatus {
                backend: "mock".into(),
                state: "ready".into(),
                llm_model: "mock".into(),
                embedding_model: "mock".into(),
                llm_available: true,
                embedding_available: true,
                last_llm_success: None,
                last_embedding_success: None,
                last_error: None,
                total_llm_calls: 0,
                total_embedding_calls: 0,
                total_prompt_tokens: 0,
                total_completion_tokens: 0,
                details: std::collections::HashMap::new(),
            }
        }

        fn batch_config(&self) -> (usize, u32) {
            (1, 0)
        }

        async fn enhance_memory_unified(
            &self,
            _prompt: &str,
        ) -> crate::error::Result<crate::llm::MemoryEnhancement> {
            Ok(crate::llm::MemoryEnhancement {
                memory_type: "Factual".into(),
                summary: "".into(),
                keywords: vec![],
                entities: vec![],
                topics: vec![],
            })
        }
    }

    // ── Wrapper tests ──

    fn make_wrapper(mock: MockClient) -> CircuitBreakerLLMClient {
        CircuitBreakerLLMClient::new(
            Box::new(mock),
            CircuitBreaker::with_config(5, 2, Duration::from_millis(50)),
            3,
            10,
            100,
            0,
        )
    }

    #[tokio::test]
    async fn test_wrapper_succeeds_on_first_try() {
        let mock = MockClient::new();
        let wrapper = make_wrapper(mock.clone());

        let result = wrapper.complete("hello").await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "mock: hello");
        assert_eq!(mock.call_count.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn test_wrapper_retries_on_failure() {
        let mock = MockClient::new();
        mock.should_fail.store(2, Ordering::Relaxed);
        let wrapper = make_wrapper(mock.clone());

        let result = wrapper.complete("hello").await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "mock: hello");
        assert_eq!(mock.call_count.load(Ordering::Relaxed), 3);
    }

    #[tokio::test]
    async fn test_wrapper_exhausts_retries() {
        let mock = MockClient::new();
        mock.should_fail.store(10, Ordering::Relaxed);
        let wrapper = make_wrapper(mock.clone());

        let result = wrapper.complete("hello").await;
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("retry attempts exhausted"));
        assert_eq!(mock.call_count.load(Ordering::Relaxed), 3);
    }

    #[tokio::test]
    async fn test_wrapper_circuit_breaker_opens() {
        let mock = MockClient::new();
        mock.should_fail.store(100, Ordering::Relaxed);
        let cb = CircuitBreaker::with_config(3, 2, Duration::from_millis(50));
        let wrapper = CircuitBreakerLLMClient::new(
            Box::new(mock.clone()),
            cb,
            1,
            10,
            100,
            0,
        );

        for _ in 0..5 {
            let _ = wrapper.complete("fail").await;
        }

        let state = wrapper.get_circuit_breaker().get_state();
        assert_eq!(state, CircuitState::Open);
    }

    #[tokio::test]
    async fn test_wrapper_circuit_breaker_recovery() {
        let mock = MockClient::new();
        mock.should_fail.store(10, Ordering::Relaxed);
        let cb = CircuitBreaker::with_config(3, 2, Duration::from_millis(50));
        let wrapper = CircuitBreakerLLMClient::new(
            Box::new(mock.clone()),
            cb,
            1,
            10,
            100,
            0,
        );

        for _ in 0..5 {
            let _ = wrapper.complete("fail").await;
        }

        assert_eq!(
            wrapper.get_circuit_breaker().get_state(),
            CircuitState::Open
        );

        mock.should_fail.store(0, Ordering::Relaxed);
        tokio::time::sleep(Duration::from_millis(60)).await;

        let result = wrapper.complete("recover").await;
        assert!(result.is_ok(), "recover call failed: {:?}", result);

        let result = wrapper.complete("recover2").await;
        assert!(result.is_ok(), "recover2 call failed: {:?}", result);

        assert_eq!(
            wrapper.get_circuit_breaker().get_state(),
            CircuitState::Closed
        );
    }

    #[tokio::test]
    async fn test_wrapper_circuit_open_rejects_immediately() {
        let mock = MockClient::new();
        mock.should_fail.store(100, Ordering::Relaxed);
        let cb = CircuitBreaker::with_config(2, 2, Duration::from_secs(10));
        let wrapper = CircuitBreakerLLMClient::new(
            Box::new(mock.clone()),
            cb,
            3,
            10,
            100,
            0,
        );

        let _ = wrapper.complete("fail1").await;
        let _ = wrapper.complete("fail2").await;

        let result = wrapper.complete("should_not_reach_inner").await;
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Circuit breaker Open"));
        assert_eq!(mock.call_count.load(Ordering::Relaxed), 2);
    }

    #[tokio::test]
    async fn test_wrapper_delegates_all_methods() {
        let mock = MockClient::new();
        let wrapper = make_wrapper(mock);

        let keywords = wrapper.extract_keywords("test content").await.unwrap();
        assert!(keywords.is_empty());

        let summary = wrapper.summarize("some content", Some(50)).await.unwrap();
        assert_eq!(summary, "");

        let health = wrapper.health_check().await.unwrap();
        assert!(health);
    }

    #[tokio::test]
    async fn test_wrapper_embed() {
        let mock = MockClient::new();
        let wrapper = make_wrapper(mock);

        let emb = wrapper.embed("test").await.unwrap();
        assert_eq!(emb, vec![1.0]);
    }

    #[tokio::test]
    async fn test_wrapper_embed_batch() {
        let mock = MockClient::new();
        let wrapper = make_wrapper(mock);

        let embs = wrapper
            .embed_batch(&["a".into(), "b".into()])
            .await
            .unwrap();
        assert_eq!(embs.len(), 2);
    }

    #[tokio::test]
    async fn test_wrapper_batch_config_passthrough() {
        let mock = MockClient::new();
        let wrapper = make_wrapper(mock);
        let (size, tokens) = wrapper.batch_config();
        assert_eq!(size, 1);
        assert_eq!(tokens, 0);
    }

    #[tokio::test]
    async fn test_wrapper_clone_shares_circuit_state() {
        let mock = MockClient::new();
        let cb = CircuitBreaker::with_config(2, 2, Duration::from_millis(50));
        let wrapper = CircuitBreakerLLMClient::new(
            Box::new(mock),
            cb.clone(),
            1,
            10,
            100,
            0,
        );

        let _ = wrapper.complete("test").await;
        let stats = cb.stats();
        assert_eq!(stats.consecutive_successes, 1);
    }

    #[tokio::test]
    async fn test_wrapper_get_status_includes_circuit_info() {
        let mock = MockClient::new();
        let wrapper = make_wrapper(mock);
        let status = wrapper.get_status();

        assert!(status.details.contains_key("circuit_breaker_state"));
        assert!(status
            .details
            .get("circuit_breaker_state")
            .unwrap()
            .as_str()
            .unwrap()
            .contains("Closed"));
        assert!(status.details.contains_key("circuit_breaker_consecutive_failures"));
        assert!(status.details.contains_key("circuit_breaker_consecutive_successes"));
    }

    #[tokio::test]
    async fn test_wrapper_with_defaults() {
        let mock = MockClient::new();
        let wrapper = CircuitBreakerLLMClient::with_defaults(Box::new(mock));

        let result = wrapper.complete("test").await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_wrapper_complete_with_grammar() {
        let mock = MockClient::new();
        let wrapper = make_wrapper(mock);

        let result = wrapper.complete_with_grammar("prompt", "grammar").await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_wrapper_classify_memory() {
        let mock = MockClient::new();
        let wrapper = make_wrapper(mock);

        let result = wrapper.classify_memory("test").await.unwrap();
        assert_eq!(result.memory_type, "Factual");
    }

    #[tokio::test]
    async fn test_wrapper_enhance_memory_unified() {
        let mock = MockClient::new();
        let wrapper = make_wrapper(mock);

        let result = wrapper.enhance_memory_unified("test").await.unwrap();
        assert_eq!(result.memory_type, "Factual");
    }

    #[tokio::test]
    async fn test_wrapper_complete_batch() {
        let mock = MockClient::new();
        let wrapper = make_wrapper(mock);

        let results = wrapper
            .complete_batch(&["a".into(), "b".into()])
            .await
            .unwrap();
        assert_eq!(results.len(), 2);
    }

    #[tokio::test]
    async fn test_wrapper_extract_metadata_enrichment_batch() {
        let mock = MockClient::new();
        let wrapper = make_wrapper(mock);

        let results = wrapper
            .extract_metadata_enrichment_batch(&["a".into(), "b".into()])
            .await
            .unwrap();
        assert_eq!(results.len(), 2);
    }

    #[tokio::test]
    async fn test_concurrent_requests_share_circuit_state() {
        let mock = MockClient::new();
        mock.should_fail.store(100, Ordering::Relaxed);
        let cb = CircuitBreaker::with_config(3, 2, Duration::from_millis(50));
        let wrapper = CircuitBreakerLLMClient::new(
            Box::new(mock.clone()),
            cb,
            1,
            10,
            100,
            0,
        );

        let mut handles = vec![];
        for i in 0..5 {
            let w = wrapper.clone();
            let handle = tokio::spawn(async move {
                let _ = w.complete(&format!("req_{}", i)).await;
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.await.unwrap();
        }

        let state = wrapper.get_circuit_breaker().get_state();
        assert_eq!(state, CircuitState::Open);
    }

    #[tokio::test]
    async fn test_circuit_breaker_reopens_after_half_open_failure() {
        let cb = CircuitBreaker::with_config(2, 2, Duration::from_millis(50));

        cb.record_failure();
        cb.record_failure();
        assert_eq!(cb.get_state(), CircuitState::Open);

        tokio::time::sleep(Duration::from_millis(60)).await;
        assert_eq!(cb.get_state(), CircuitState::HalfOpen);

        cb.record_success();
        cb.record_failure();
        assert_eq!(cb.get_state(), CircuitState::Open);

        tokio::time::sleep(Duration::from_millis(60)).await;
        assert_eq!(cb.get_state(), CircuitState::HalfOpen);

        cb.record_success();
        cb.record_success();
        assert_eq!(cb.get_state(), CircuitState::Closed);
    }

    #[tokio::test]
    async fn test_backoff_increases_delay_between_retries() {
        let mock = MockClient::new();
        mock.should_fail.store(10, Ordering::Relaxed);
        let wrapper = CircuitBreakerLLMClient::new(
            Box::new(mock),
            CircuitBreaker::with_config(100, 2, Duration::from_secs(60)),
            3,
            50,
            10_000,
            0,
        );

        let start = std::time::Instant::now();
        let _ = wrapper.complete("test").await;
        let elapsed = start.elapsed();

        assert!(
            elapsed >= Duration::from_millis(100),
            "Expected at least 100ms (50+100) for two backoff delays, got {:?}",
            elapsed
        );
    }

    #[test]
    fn test_circuit_breaker_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<CircuitBreaker>();
    }

    #[test]
    fn test_circuit_breaker_stats_after_open() {
        let cb = CircuitBreaker::with_config(2, 2, Duration::from_millis(50));
        cb.record_failure();
        cb.record_failure();

        let stats = cb.stats();
        assert_eq!(stats.state, CircuitState::Open);
        assert_eq!(stats.consecutive_failures, 2);
    }
}
