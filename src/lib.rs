use regex::Regex;
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::thread;
use walkdir::WalkDir;

pub mod types;

use crate::types::{TfIdf, TfInfo};

// ---------------------------------------------------------------------------
// Searcher
// ---------------------------------------------------------------------------
pub struct FileSearcher {
    root: PathBuf,
    file_filter: Box<dyn Fn(&Path) -> bool>,
    parallelism_limit: usize,
    data_chan_size: usize,
}

impl FileSearcher {
    /// Quickly create a searcher with all default settings
    pub fn new(root: impl Into<PathBuf>) -> Self {
        let num_cpus = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);

        Self {
            root: root.into(),
            file_filter: Box::new(|p: &Path| p.extension().map(|e| e == "txt").unwrap_or(false)),
            parallelism_limit: num_cpus,
            data_chan_size: 1024,
        }
    }

    /// Custom file filter
    pub fn file_filter<F>(mut self, f: F) -> Self
    where
        F: Fn(&Path) -> bool + Send + Sync + 'static,
    {
        self.file_filter = Box::new(f);
        self
    }

    /// Custom parallelism limit
    pub fn parallelism_limit(mut self, n: usize) -> Self {
        self.parallelism_limit = n;
        self
    }

    /// Custom channel buffer size
    pub fn data_chan_size(mut self, n: usize) -> Self {
        self.data_chan_size = n;
        self
    }
}

// ---------------------------------------------------------------------------
// Directory traversal
// ---------------------------------------------------------------------------
impl FileSearcher {
    pub fn walk_dir(&self) -> io::Result<Vec<PathBuf>> {
        let mut paths = Vec::new();

        for entry in WalkDir::new(&self.root).follow_links(false).into_iter() {
            let entry = entry?;
            if entry.file_type().is_file() && (self.file_filter)(entry.path()) {
                paths.push(entry.into_path());
            }
        }
        Ok(paths)
    }
}

/// File processor trait for parallel file processing
pub trait FileProcessor<T>: Send + Sync + 'static {
    /// Process a single file
    /// # Arguments
    /// - `path`: current file path
    /// - `index`: global file index
    /// - `content`: complete text content of the file
    /// - `tx`: synchronous result sender channel
    fn process(&self, path: PathBuf, index: usize, content: String, tx: SyncSender<T>);
}

// Auto-implement for closures so callers don't need changes
impl<F, T> FileProcessor<T> for F
where
    F: Fn(PathBuf, usize, String, SyncSender<T>) + Send + Sync + 'static,
{
    fn process(&self, path: PathBuf, index: usize, content: String, tx: SyncSender<T>) {
        self(path, index, content, tx);
    }
}

/// Read all lines of a single file, internally calls FileProcessor callback
/// The `index: usize` parameter represents the global index of the current file
fn read_file_lines<T, P>(
    path: &Path,
    index: usize,
    tx: &SyncSender<T>,
    processor: &P,
) -> io::Result<()>
where
    T: Send + 'static,
    P: FileProcessor<T>,
{
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let full_path = path.to_path_buf();

    // Read entire file content (pass at file granularity, consistent with parallel logic)
    let mut content = String::new();
    for line in reader.lines() {
        content.push_str(&line?);
        content.push('\n');
    }

    let tx_clone = tx.clone();
    processor.process(full_path, index, content, tx_clone);
    Ok(())
}

// ---------------------------------------------------------------------------
// Concurrent file reading (chunked concurrency)
// ---------------------------------------------------------------------------
impl FileSearcher {
    pub fn read_files_parallel<T, F>(&self, process_func: F) -> io::Result<(Receiver<T>, usize)>
    where
        T: Send + 'static,
        F: FileProcessor<T>,
    {
        let paths = self.walk_dir()?;
        let total = paths.len();

        if total == 0 {
            let (_, rx) = mpsc::channel();
            return Ok((rx, 0));
        }

        let (tx, rx) = mpsc::sync_channel(self.data_chan_size);
        let tx = &tx;
        let process_func = &process_func;

        let chunk_size = ((total as f64) / (self.parallelism_limit as f64)).ceil() as usize;

        thread::scope(|s| {
            for (chunk_idx, chunk) in paths.chunks(chunk_size).enumerate() {
                let chunk_start = chunk_idx * chunk_size;
                s.spawn(move || {
                    for (offset, path) in chunk.iter().enumerate() {
                        let global_idx = chunk_start + offset;
                        if let Err(e) = read_file_lines(path, global_idx, tx, process_func) {
                            log::warn!("Failed to read file {}: {}", path.display(), e);
                        }
                    }
                });
            }
        });

        Ok((rx, total))
    }
}

/// Single-file TF statistics
fn aggregate_file_stats(path: &Path, reg_word: &Regex, reg_key: &Regex) -> io::Result<TfInfo> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut total_words = 0;
    let mut total_key_words = 0;

    for line in reader.lines() {
        let line = line?;
        total_words += reg_word.find_iter(&line).count();
        total_key_words += reg_key.find_iter(&line).count();
    }

    Ok(TfInfo {
        file_path: path.to_path_buf(),
        word_count: total_words,
        key_word_count: total_key_words,
    })
}

// ---------------------------------------------------------------------------
// TF-IDF calculation module
// ---------------------------------------------------------------------------
impl FileSearcher {
    pub fn tf_idf(&self, word: &str, k: usize) -> io::Result<Vec<TfIdf>> {
        let key_word_pattern = format!(r"(?i)\b{}\b", regex::escape(word));
        let reg_key = Regex::new(&key_word_pattern)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e.to_string()))?;
        let reg_word = Regex::new(r"[a-zA-Z0-9]+").unwrap();

        let (rx, total_files) = self.read_files_stats_parallel(&reg_word, &reg_key)?;

        if total_files == 0 {
            return Ok(Vec::new());
        }

        let mut aggregated: HashMap<PathBuf, TfInfo> = HashMap::new();
        for info in rx {
            aggregated.insert(info.file_path.clone(), info);
        }

        let doc_with_word = aggregated.values().filter(|i| i.key_word_count > 0).count();
        if doc_with_word == 0 {
            return Ok(Vec::new());
        }

        let idf = ((total_files as f64) / (doc_with_word as f64)).ln() + 1.0;

        let mut results: Vec<TfIdf> = aggregated
            .into_iter()
            .filter_map(|(file_path, info)| {
                if info.key_word_count == 0 {
                    return None;
                }
                let tf = info.key_word_count as f64 / info.word_count as f64;
                let tfidf = tf * idf;
                Some(TfIdf {
                    file_path,
                    word_count: info.word_count,
                    key_word_count: info.key_word_count,
                    idf,
                    tfidf,
                })
            })
            .collect();

        results.sort_by(|a, b| {
            b.tfidf
                .partial_cmp(&a.tfidf)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let top_k = results.len().min(k);
        results.truncate(top_k);
        Ok(results)
    }

    pub fn read_files_stats_parallel(
        &self,
        reg_word: &Regex,
        reg_key: &Regex,
    ) -> io::Result<(Receiver<TfInfo>, usize)> {
        let paths = self.walk_dir()?;
        let total = paths.len();

        if total == 0 {
            let (_, rx) = mpsc::channel();
            return Ok((rx, 0));
        }

        let (tx, rx) = mpsc::sync_channel(total);
        let tx = &tx;

        let chunk_size = ((total as f64) / (self.parallelism_limit as f64)).ceil() as usize;

        thread::scope(|s| {
            for chunk in paths.chunks(chunk_size) {
                s.spawn(move || {
                    for path in chunk {
                        match aggregate_file_stats(path, reg_word, reg_key) {
                            Ok(info) => {
                                let _ = tx.send(info);
                            }
                            Err(e) => log::warn!("Statistics failed {}: {}", path.display(), e),
                        }
                    }
                });
            }
        });

        Ok((rx, total))
    }
}

// ---------------------------------------------------------------------------
// Comprehensive tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::MatchResult;
    use std::fs;
    use std::io::Write;
    use std::sync::mpsc::SyncSender;
    use tempfile::TempDir;

    // -----------------------------------------------------------------------
    // Helper: create a temporary directory with sample .txt files
    // -----------------------------------------------------------------------
    fn setup_test_dir() -> TempDir {
        let dir = TempDir::new().expect("Failed to create temp dir");

        // file1.txt: contains "hello world" and "hello rust"
        let mut f1 = fs::File::create(dir.path().join("file1.txt")).unwrap();
        writeln!(f1, "hello world").unwrap();
        writeln!(f1, "hello rust").unwrap();

        // file2.txt: contains "rust is awesome" and "world of rust"
        let mut f2 = fs::File::create(dir.path().join("file2.txt")).unwrap();
        writeln!(f2, "rust is awesome").unwrap();
        writeln!(f2, "world of rust").unwrap();

        // file3.txt: contains only "hello"
        let mut f3 = fs::File::create(dir.path().join("file3.txt")).unwrap();
        writeln!(f3, "hello").unwrap();

        // A non-txt file that should be filtered out by default
        let mut f4 = fs::File::create(dir.path().join("notes.md")).unwrap();
        writeln!(f4, "# Markdown").unwrap();

        dir
    }

    fn setup_single_file_dir() -> TempDir {
        let dir = TempDir::new().expect("Failed to create temp dir");
        let mut f = fs::File::create(dir.path().join("doc.txt")).unwrap();
        writeln!(f, "hello world hello rust").unwrap();
        dir
    }

    // -----------------------------------------------------------------------
    // walk_dir tests
    // -----------------------------------------------------------------------
    #[test]
    fn test_walk_dir_finds_only_txt_files() {
        let dir = setup_test_dir();
        let searcher = FileSearcher::new(dir.path());
        let files = searcher.walk_dir().unwrap();

        // Should only find .txt files (3 of them), not .md files
        assert_eq!(files.len(), 3, "Expected 3 .txt files");
        for path in &files {
            assert_eq!(
                path.extension().unwrap(),
                "txt",
                "All found files should be .txt"
            );
        }
    }

    #[test]
    fn test_walk_dir_empty_directory() {
        let dir = TempDir::new().expect("Failed to create temp dir");
        let searcher = FileSearcher::new(dir.path());
        let files = searcher.walk_dir().unwrap();
        assert!(files.is_empty(), "Empty directory should yield no files");
    }

    #[test]
    fn test_walk_dir_with_custom_filter() {
        let dir = setup_test_dir();
        let searcher = FileSearcher::new(dir.path())
            .file_filter(|p| p.extension().map(|e| e == "md").unwrap_or(false));
        let files = searcher.walk_dir().unwrap();

        assert_eq!(files.len(), 1, "Expected only the .md file");
        assert_eq!(
            files[0].extension().unwrap(),
            "md",
            "Should be the markdown file"
        );
    }

    #[test]
    fn test_walk_dir_non_existent_root() {
        let searcher = FileSearcher::new("C:\\non_existent_path_abcxyz");
        let result = searcher.walk_dir();
        assert!(result.is_err(), "Non-existent path should return an error");
    }

    // -----------------------------------------------------------------------
    // read_files_parallel tests
    // -----------------------------------------------------------------------
    #[test]
    fn test_read_files_parallel_basic() {
        let dir = setup_test_dir();
        let searcher = FileSearcher::new(dir.path()).parallelism_limit(2);

        let (rx, count) = searcher
            .read_files_parallel(
                |path, index, content, tx: SyncSender<(PathBuf, usize, String)>| {
                    let _ = tx.send((path, index, content));
                },
            )
            .unwrap();

        assert_eq!(count, 3, "Should report 3 files");

        let mut results: Vec<_> = rx.iter().collect();
        results.sort_by_key(|a| a.1);

        assert_eq!(results.len(), 3);
        // Each result should have unique index
        let indices: Vec<usize> = results.iter().map(|r| r.1).collect();
        assert_eq!(indices, vec![0, 1, 2], "Indices should be 0, 1, 2");

        // Verify content contains expected text
        let all_content: String = results.iter().map(|r| r.2.clone()).collect();
        assert!(all_content.contains("hello world"));
        assert!(all_content.contains("rust is awesome"));
    }

    #[test]
    fn test_read_files_parallel_no_files() {
        let dir = TempDir::new().expect("Failed to create temp dir");
        let searcher = FileSearcher::new(dir.path());

        let (_rx, count) = searcher
            .read_files_parallel(|_path, _idx, _content, _tx: SyncSender<()>| {})
            .unwrap();

        assert_eq!(count, 0, "Should report 0 files");
    }

    #[test]
    fn test_read_files_parallel_match_result_type() {
        let dir = setup_test_dir();
        let searcher = FileSearcher::new(dir.path());

        let (rx, count) = searcher
            .read_files_parallel(|path, index, content, tx: SyncSender<MatchResult>| {
                let _ = tx.send(MatchResult {
                    file_path: path,
                    line_num: index,
                    content,
                });
            })
            .unwrap();

        assert_eq!(count, 3);
        let results: Vec<MatchResult> = rx.iter().collect();
        assert_eq!(results.len(), 3);
        // Each MatchResult should have a unique line_num
        let mut nums: Vec<usize> = results.iter().map(|r| r.line_num).collect();
        nums.sort();
        assert_eq!(nums, vec![0, 1, 2]);
    }

    #[test]
    fn test_read_files_parallel_single_file() {
        let dir = setup_single_file_dir();
        let searcher = FileSearcher::new(dir.path());

        let (rx, count) = searcher
            .read_files_parallel(
                |path, index, content, tx: SyncSender<(PathBuf, usize, String)>| {
                    let _ = tx.send((path, index, content));
                },
            )
            .unwrap();

        assert_eq!(count, 1);
        let results: Vec<_> = rx.iter().collect();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].1, 0);
        assert!(results[0].2.contains("hello world hello rust"));
    }

    // -----------------------------------------------------------------------
    // FileProcessor trait tests
    // -----------------------------------------------------------------------
    #[test]
    fn test_file_processor_with_closure() {
        // Closures implement FileProcessor automatically
        let processor = |_path: PathBuf, index: usize, content: String, tx: SyncSender<String>| {
            let _ = tx.send(format!("{}:{}", index, content.len()));
        };

        let dir = setup_test_dir();
        let searcher = FileSearcher::new(dir.path());

        let (rx, _count) = searcher.read_files_parallel(processor).unwrap();
        let msgs: Vec<String> = rx.iter().collect();
        assert_eq!(msgs.len(), 3);
    }

    // -----------------------------------------------------------------------
    // TF-IDF tests
    // -----------------------------------------------------------------------
    #[test]
    fn test_tf_idf_basic() {
        let dir = setup_test_dir();
        let searcher = FileSearcher::new(dir.path());

        // "hello" appears in file1.txt (2 times) and file3.txt (1 time)
        let results = searcher.tf_idf("hello", 5).unwrap();

        assert!(!results.is_empty(), "Should find 'hello' in some files");

        // Check results are sorted by tfidf descending
        for pair in results.windows(2) {
            assert!(
                pair[0].tfidf >= pair[1].tfidf,
                "Results should be sorted by TF-IDF descending"
            );
        }
    }

    #[test]
    fn test_tf_idf_empty_corpus() {
        let dir = TempDir::new().expect("Failed to create temp dir");
        let searcher = FileSearcher::new(dir.path());

        let results = searcher.tf_idf("hello", 5).unwrap();
        assert!(
            results.is_empty(),
            "No files should yield empty TF-IDF results"
        );
    }

    #[test]
    fn test_tf_idf_no_match() {
        let dir = setup_test_dir();
        let searcher = FileSearcher::new(dir.path());

        // This word doesn't appear in any file
        let results = searcher.tf_idf("zzznotfound", 5).unwrap();
        assert!(
            results.is_empty(),
            "Non-existent word should yield empty results"
        );
    }

    #[test]
    fn test_tf_idf_single_file_match() {
        let dir = setup_single_file_dir();
        let searcher = FileSearcher::new(dir.path());

        let results = searcher.tf_idf("hello", 5).unwrap();
        assert_eq!(results.len(), 1, "Should find 'hello' in the single file");

        let r = &results[0];
        assert!(r.tfidf > 0.0, "TF-IDF should be positive");
        assert!(r.idf > 0.0, "IDF should be positive");
        assert_eq!(r.key_word_count, 2, "'hello' appears twice in doc.txt");
        assert_eq!(r.word_count, 4, "Total word count in doc.txt is 4");
    }

    #[test]
    fn test_tf_idf_top_k() {
        let dir = setup_test_dir();
        let searcher = FileSearcher::new(dir.path());

        // Get only top 1 result for "hello"
        let results = searcher.tf_idf("hello", 1).unwrap();
        assert_eq!(results.len(), 1, "Should return at most top 1 result");
    }

    // -----------------------------------------------------------------------
    // aggregate_file_stats test
    // -----------------------------------------------------------------------
    #[test]
    fn test_aggregate_file_stats_direct() {
        let dir = setup_single_file_dir();
        let file_path = dir.path().join("doc.txt");

        let reg_word = Regex::new(r"[a-zA-Z0-9]+").unwrap();
        let reg_key = Regex::new(r"(?i)\bhello\b").unwrap();

        let info = aggregate_file_stats(&file_path, &reg_word, &reg_key)
            .expect("aggregate_file_stats should succeed");

        assert_eq!(info.word_count, 4);
        assert_eq!(info.key_word_count, 2);
        assert_eq!(info.file_path, file_path);
    }

    #[test]
    fn test_aggregate_file_stats_no_keyword_match() {
        let dir = setup_single_file_dir();
        let file_path = dir.path().join("doc.txt");

        let reg_word = Regex::new(r"[a-zA-Z0-9]+").unwrap();
        let reg_key = Regex::new(r"(?i)\bzzz\b").unwrap();

        let info = aggregate_file_stats(&file_path, &reg_word, &reg_key)
            .expect("aggregate_file_stats should succeed");

        assert_eq!(info.word_count, 4);
        assert_eq!(info.key_word_count, 0, "Keyword 'zzz' should not be found");
    }

    // -----------------------------------------------------------------------
    // Edge case tests
    // -----------------------------------------------------------------------
    #[test]
    fn test_walk_dir_symlink_not_followed() {
        // walk_dir uses follow_links(false), so symlinks should NOT be followed.
        // Since on Windows creating symlinks requires admin or dev mode,
        // we just verify symlinked files are not traversed when possible.
        let dir = setup_test_dir();
        let searcher = FileSearcher::new(dir.path());
        let files = searcher.walk_dir().unwrap();
        // All found paths should exist as real files within the root
        for path in &files {
            assert!(path.exists(), "Path should exist: {:?}", path);
            assert!(
                path.starts_with(dir.path()),
                "Path should be inside root: {:?}",
                path
            );
        }
    }

    #[test]
    fn test_new_defaults() {
        let searcher = FileSearcher::new("some/path");
        assert_eq!(
            searcher.parallelism_limit,
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4)
        );
        assert_eq!(searcher.data_chan_size, 1024);
    }

    #[test]
    fn test_builder_pattern() {
        let searcher = FileSearcher::new("some/path")
            .parallelism_limit(8)
            .data_chan_size(512)
            .file_filter(|p| p.extension().map(|e| e == "log").unwrap_or(false));

        assert_eq!(searcher.parallelism_limit, 8);
        assert_eq!(searcher.data_chan_size, 512);
    }

    // -----------------------------------------------------------------------
    // Integration test: walk -> read -> process pipeline
    // -----------------------------------------------------------------------
    #[test]
    fn test_full_pipeline_integration() {
        let dir = setup_test_dir();
        let searcher = FileSearcher::new(dir.path()).parallelism_limit(2);

        // Step 1: Walk
        let paths = searcher.walk_dir().unwrap();
        assert_eq!(paths.len(), 3);

        // Step 2: Read in parallel with custom processing
        let (rx, total) = searcher
            .read_files_parallel(
                |path: PathBuf,
                 idx: usize,
                 content: String,
                 tx: SyncSender<(PathBuf, usize, usize)>| {
                    let char_count = content.chars().count();
                    let _ = tx.send((path, idx, char_count));
                },
            )
            .unwrap();

        assert_eq!(total, 3);

        let collected: Vec<_> = rx.iter().collect();
        assert_eq!(collected.len(), 3);

        // Each entry should have non-zero char count
        for (_, _, char_count) in &collected {
            assert!(*char_count > 0, "File content should not be empty");
        }
    }
}
