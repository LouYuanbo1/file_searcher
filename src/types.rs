use std::path::PathBuf;
// ---------------------------------------------------------------------------
// Search result / TF-IDF result types
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct MatchResult {
    pub file_path: PathBuf,
    pub line_num: usize,
    pub content: String,
}

/// Per-file statistics
#[derive(Debug, Clone)]
pub struct TfInfo {
    pub file_path: PathBuf,
    pub word_count: usize,
    pub key_word_count: usize,
}

/// Final TF-IDF result
#[derive(Debug, Clone)]
pub struct TfIdf {
    pub file_path: PathBuf,
    pub word_count: usize,
    pub key_word_count: usize,
    pub idf: f64,
    pub tfidf: f64,
}
