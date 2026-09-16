# file_searcher

A concurrent file searcher written in Rust that supports parallel file traversal, custom filtering, and TF-IDF based relevance ranking.

## Features

- **Concurrent file reading** — Automatically distributes files across multiple threads using scoped threads and chunked work distribution.
- **Custom file filtering** — Filter files by extension, name pattern, or any custom predicate.
- **Parallel file processing** — Process file contents concurrently with a flexible `FileProcessor` trait (closures work out of the box).
- **TF-IDF ranking** — Compute Term Frequency-Inverse Document Frequency scores for a given keyword across a corpus of text files, returning results sorted by relevance.

## Quick Start

Add `file_searcher` to your `Cargo.toml`:

```toml
[dependencies]
file_searcher = { git = "https://github.com/LouYuanbo1/file_searcher" }
```

### Basic Usage

```rust
use file_searcher::FileSearcher;
use std::sync::mpsc::SyncSender;

// Create a searcher rooted at a directory (default: only .txt files)
let searcher = FileSearcher::new("path/to/corpus")
    .parallelism_limit(4)
    .data_chan_size(1024);

// Walk the directory and collect file paths
let files = searcher.walk_dir().unwrap();
println!("Found {} files", files.len());

// Read files in parallel with a custom processor
let (rx, total) = searcher
    .read_files_parallel(
        |path: std::path::PathBuf,
         index: usize,
         content: String,
         tx: SyncSender<(std::path::PathBuf, usize)>| {
            let line_count = content.lines().count();
            let _ = tx.send((path, line_count));
        },
    )
    .unwrap();

for (path, line_count) in rx {
    println!("{}: {} lines", path.display(), line_count);
}
```

### TF-IDF Search

```rust
use file_searcher::FileSearcher;

let searcher = FileSearcher::new("path/to/corpus");

// Search for the top-5 most relevant documents for "rust"
let results = searcher.tf_idf("rust", 5).unwrap();

for r in &results {
    println!(
        "{}  TF={:.4}  IDF={:.4}  TF-IDF={:.4}",
        r.file_path.display(),
        r.key_word_count as f64 / r.word_count as f64,
        r.idf,
        r.tfidf
    );
}
```

## API Overview

### `FileSearcher`

The main entry point. Configure it with a builder pattern:

| Method | Default | Description |
|---|---|---|
| `new(root)` | — | Create a searcher rooted at `root`. Default filter: `*.txt` files. Parallelism: number of logical CPUs. Buffer size: 1024. |
| `file_filter(f)` | `\|p\| p.is_txt()` | Set a custom filter function `Fn(&Path) -> bool` |
| `parallelism_limit(n)` | CPU count | Set the number of worker threads |
| `data_chan_size(n)` | 1024 | Set the channel buffer size between threads |

### Methods

- **`walk_dir()`** → `io::Result<Vec<PathBuf>>` — Recursively walk the root directory and collect matching file paths.
- **`read_files_parallel<F>(process_func)`** → `io::Result<(Receiver<T>, usize)>` — Walk files and process each one concurrently via the `FileProcessor` trait. Returns a channel receiver and total file count.
- **`tf_idf(word, k)`** → `io::Result<Vec<TfIdf>>` — Compute TF-IDF scores for `word` across all matching files, returning the top `k` results sorted by relevance.
- **`read_files_stats_parallel(reg_word, reg_key)`** → `io::Result<(Receiver<TfInfo>, usize)>` — Walk files and compute per-file word/keyword counts in parallel.

### Key Types

| Type | Fields | Description |
|---|---|---|
| `MatchResult` | `file_path`, `line_num`, `content` | A generic file match result |
| `TfInfo` | `file_path`, `word_count`, `key_word_count` | Per-file word and keyword counts |
| `TfIdf` | `file_path`, `word_count`, `key_word_count`, `idf`, `tfidf` | TF-IDF score for a document |

### `FileProcessor<T>` Trait

Implemented automatically for any closure matching `Fn(PathBuf, usize, String, SyncSender<T>)`. Custom structs can implement it directly for more complex processing logic.

## Testing

Run the full test suite:

```bash
cargo test
```

The test suite includes **20 tests** covering:

- Directory walking with default and custom filters
- Empty directories and non-existent paths
- Parallel file reading with various data types
- TF-IDF computation (basic, empty corpus, no-match, single file, top-k truncation)
- `aggregate_file_stats` direct unit tests
- Builder pattern and default values
- Full pipeline integration test

## License

This project is licensed under the MIT License.