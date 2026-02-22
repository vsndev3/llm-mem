use thiserror::Error;

#[derive(Error, Debug)]
pub enum MemoryError {
    #[error("VectorLite error: {0}")]
    VectorLite(String),

    #[error("LLM error: {0}")]
    LLM(String),

    #[error("Local inference error: {0}")]
    LocalInference(String),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Memory not found: {id}")]
    NotFound { id: String },

    #[error("Invalid memory action: {action}")]
    InvalidAction { action: String },

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Validation error: {0}")]
    Validation(String),

    #[error("Embedding error: {0}")]
    Embedding(String),

    #[error("Parse error: {0}")]
    Parse(String),

    #[error("Download error: {0}")]
    Download(String),
}

pub type Result<T> = std::result::Result<T, MemoryError>;

impl MemoryError {
    pub fn config<S: Into<String>>(msg: S) -> Self {
        Self::Config(msg.into())
    }

    pub fn validation<S: Into<String>>(msg: S) -> Self {
        Self::Validation(msg.into())
    }

    pub fn embedding<S: Into<String>>(msg: S) -> Self {
        Self::Embedding(msg.into())
    }

    pub fn parse<S: Into<String>>(msg: S) -> Self {
        Self::Parse(msg.into())
    }

    pub fn download<S: Into<String>>(msg: S) -> Self {
        Self::Download(msg.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vectorlite_error_display() {
        let err = MemoryError::VectorLite("index failed".to_string());
        assert_eq!(err.to_string(), "VectorLite error: index failed");
    }

    #[test]
    fn test_llm_error_display() {
        let err = MemoryError::LLM("timeout".to_string());
        assert_eq!(err.to_string(), "LLM error: timeout");
    }

    #[test]
    fn test_not_found_error_display() {
        let err = MemoryError::NotFound {
            id: "abc-123".to_string(),
        };
        assert_eq!(err.to_string(), "Memory not found: abc-123");
    }

    #[test]
    fn test_invalid_action_error_display() {
        let err = MemoryError::InvalidAction {
            action: "fly".to_string(),
        };
        assert_eq!(err.to_string(), "Invalid memory action: fly");
    }

    #[test]
    fn test_config_convenience_constructor() {
        let err = MemoryError::config("bad setting");
        assert_eq!(err.to_string(), "Configuration error: bad setting");
    }

    #[test]
    fn test_validation_convenience_constructor() {
        let err = MemoryError::validation("field missing");
        assert_eq!(err.to_string(), "Validation error: field missing");
    }

    #[test]
    fn test_embedding_convenience_constructor() {
        let err = MemoryError::embedding("dimension mismatch");
        assert_eq!(err.to_string(), "Embedding error: dimension mismatch");
    }

    #[test]
    fn test_parse_convenience_constructor() {
        let err = MemoryError::parse("unexpected token");
        assert_eq!(err.to_string(), "Parse error: unexpected token");
    }

    #[test]
    fn test_serialization_error_from_serde() {
        let json_err = serde_json::from_str::<serde_json::Value>("not valid json").unwrap_err();
        let err: MemoryError = json_err.into();
        assert!(err.to_string().contains("Serialization error"));
    }

    #[test]
    fn test_error_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        // MemoryError should be Send + Sync
        assert_send_sync::<MemoryError>();
    }

    #[test]
    fn test_error_debug_impl() {
        let err = MemoryError::config("test");
        let debug_str = format!("{:?}", err);
        assert!(debug_str.contains("Config"));
    }

    #[test]
    fn test_local_inference_error_display() {
        let err = MemoryError::LocalInference("model load failed".to_string());
        assert_eq!(err.to_string(), "Local inference error: model load failed");
    }

    #[test]
    fn test_embedding_error_display() {
        let err = MemoryError::Embedding("dimension mismatch".to_string());
        assert_eq!(err.to_string(), "Embedding error: dimension mismatch");
    }

    #[test]
    fn test_parse_error_display() {
        let err = MemoryError::Parse("invalid JSON in model output".to_string());
        assert_eq!(err.to_string(), "Parse error: invalid JSON in model output");
    }

    #[test]
    fn test_download_error_display() {
        let err = MemoryError::Download("connection refused".to_string());
        assert_eq!(err.to_string(), "Download error: connection refused");
    }

    #[test]
    fn test_download_convenience_constructor() {
        let err = MemoryError::download("network timeout");
        assert_eq!(err.to_string(), "Download error: network timeout");
    }

    #[test]
    fn test_download_error_with_multiline_message() {
        let err = MemoryError::download(
            "Failed to connect to huggingface.co\n\nPlease check your proxy settings.",
        );
        let msg = err.to_string();
        assert!(msg.contains("Failed to connect"));
        assert!(msg.contains("proxy settings"));
    }
}
