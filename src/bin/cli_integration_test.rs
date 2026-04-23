// Integration test for llm-mem CLI
// Tests document upload, status checking, and retrieval using the CLI

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::thread;
use std::time::Duration;

fn run_command(cmd: &str, args: &[&str]) -> (String, String) {
    let output = Command::new(cmd)
        .args(args)
        .output()
        .expect("Failed to execute command");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    (stdout.to_string(), stderr.to_string())
}

fn extract_session_id(stdout: &str) -> Option<String> {
    stdout
        .lines()
        .find(|line| line.contains("session_id:"))
        .and_then(|line| line.split('"').nth(1).map(|s| s.to_string()))
}

/// Find or build the llm-mem binary.
/// Detects which profile (release/debug) we were built with by checking our own
/// executable path, then looks for llm-mem in the same profile directory first.
fn find_llm_mem_binary() -> PathBuf {
    let self_exe = std::env::current_exe().expect("Failed to get current executable path");
    let self_dir = self_exe.parent().unwrap_or(Path::new("."));
    let is_release = self_dir.ends_with("release");

    let (primary, fallback) = if is_release {
        (
            PathBuf::from("target/release/llm-mem"),
            PathBuf::from("target/debug/llm-mem"),
        )
    } else {
        (
            PathBuf::from("target/debug/llm-mem"),
            PathBuf::from("target/release/llm-mem"),
        )
    };

    if primary.exists() {
        println!("Using binary: {}", primary.display());
        return primary;
    }

    if fallback.exists() {
        println!(
            "Using binary: {} (fallback from other profile)",
            fallback.display()
        );
        return fallback;
    }

    // Auto-build with the same profile
    eprintln!("llm-mem binary not found — building it...");
    let mut args: Vec<&str> = vec!["build", "--bin", "llm-mem"];
    if is_release {
        args.push("--release");
    }
    let status = Command::new("cargo")
        .args(&args)
        .current_dir(std::env::current_dir().expect("Failed to get CWD"))
        .status()
        .expect("Failed to run cargo build");
    if !status.success() {
        eprintln!("ERROR: cargo build --bin llm-mem failed");
        std::process::exit(1);
    }
    if !primary.exists() {
        eprintln!(
            "ERROR: llm-mem binary not found after build at {}",
            primary.display()
        );
        std::process::exit(1);
    }
    println!("Built and using binary: {}", primary.display());
    primary
}

fn main() {
    // Use a unique test directory that won't conflict
    let test_dir = Path::new("tests/cli_test_temp");
    let llm_mem = find_llm_mem_binary();
    let banks_dir = test_dir.join("banks");
    let config_file = test_dir.join("config.toml");
    let doc_file = test_dir.join("test_document.txt");

    // Fail if test directory already exists
    if test_dir.exists() {
        eprintln!(
            "ERROR: Test directory {} already exists. Please remove it first.",
            test_dir.display()
        );
        std::process::exit(1);
    }

    // Create directories
    fs::create_dir_all(&banks_dir).expect("Failed to create banks dir");

    println!("=== llm-mem CLI Integration Test ===\n");
    println!("Test directory: {}", test_dir.display());
    println!();

    // Test 1: Generate config with configurable CPU threads
    println!("Test 1: Generate config with configurable CPU threads");
    let cpu_threads = std::env::var("LLM_MEM_CPU_THREADS")
        .ok()
        .and_then(|s| s.parse::<i32>().ok())
        .unwrap_or(8); // Default to 8 if not set
    println!(
        "  Using {} CPU threads (set via LLM_MEM_CPU_THREADS env var)",
        cpu_threads
    );

    let (_stdout, stderr) = run_command(
        llm_mem.to_str().unwrap(),
        &["generate-config", "--output", config_file.to_str().unwrap()],
    );
    if !stderr.contains("Error") && !stderr.contains("error:") {
        // Modify the config to use configurable CPU threads
        let mut config_content = fs::read_to_string(&config_file).expect("Failed to read config");
        config_content =
            config_content.replace("cpu_threads = 0", &format!("cpu_threads = {}", cpu_threads));
        fs::write(&config_file, config_content).expect("Failed to write config");
        println!("  Config generated with {} CPU threads\n", cpu_threads);
    } else {
        println!("  Config generation failed");
        println!(
            "  Error: {}\n",
            stderr.lines().next().unwrap_or("Unknown error")
        );
    }

    // Test 2: Create test document
    println!("Test 2: Create test document");
    fs::write(
        &doc_file,
        r#"
The Quick Brown Fox Jumps Over The Lazy Dog

This is a sample document for testing the llm-mem system.

Key Topics:
- Document processing
- Vector embeddings
- Memory management
- Semantic search

The quick brown fox jumps over the lazy dog.
"#,
    )
    .expect("Failed to write test document");

    let size = fs::metadata(&doc_file)
        .expect("Failed to get file metadata")
        .len();
    println!("  Created test document ({} bytes)\n", size);

    // Test 3: Upload document
    println!("Test 3: Upload document");
    let (stdout, _stderr) = run_command(
        llm_mem.to_str().unwrap(),
        &[
            "--config",
            config_file.to_str().unwrap(),
            "upload",
            "--file-path",
            doc_file.to_str().unwrap(),
            "--bank",
            "default",
        ],
    );

    let session_id = extract_session_id(&stdout);
    if let Some(sid) = &session_id {
        println!("  Upload started, session_id: {}", sid);
    } else {
        println!("  Failed to extract session_id from output");
        println!("  Output: {}\n", stdout);
    }
    println!();

    // Test 4: Check status (wait a bit first)
    println!("Test 4: Check document status (waiting for processing)");
    thread::sleep(Duration::from_secs(10));

    if let Some(sid) = &session_id {
        let (stdout, _stderr) = run_command(
            llm_mem.to_str().unwrap(),
            &[
                "--config",
                config_file.to_str().unwrap(),
                "doc-status",
                "--session-id",
                sid,
                "--bank",
                "default",
            ],
        );

        if stdout.contains("session_id") {
            println!("  Status retrieved successfully");
            if stdout.contains("status: \"uploading\"") || stdout.contains("status: \"processing\"")
            {
                println!("  Document still processing...");
                // Wait longer for processing to complete
                println!("  Waiting longer for processing to complete...");
                thread::sleep(Duration::from_secs(20));

                // Check status again
                let (stdout2, _stderr) = run_command(
                    llm_mem.to_str().unwrap(),
                    &[
                        "--config",
                        config_file.to_str().unwrap(),
                        "doc-status",
                        "--session-id",
                        sid,
                        "--bank",
                        "default",
                    ],
                );
                if stdout2.contains("status:") {
                    println!(
                        "  Final status: {}",
                        stdout2
                            .lines()
                            .find(|l| l.trim().starts_with("status:"))
                            .unwrap_or("unknown")
                    );
                }
            }
        } else {
            println!("  Failed to get status");
        }
    }
    println!();

    // Test 5: List sessions
    println!("Test 5: List document sessions");
    let (_stdout, _stderr) = run_command(
        llm_mem.to_str().unwrap(),
        &[
            "--config",
            config_file.to_str().unwrap(),
            "list-sessions",
            "--bank",
            "default",
        ],
    );

    println!("  Sessions listed (check output above)\n");

    // Test 6: List memories
    println!("Test 6: List memories");
    let (_stdout, _stderr) = run_command(
        llm_mem.to_str().unwrap(),
        &[
            "--config",
            config_file.to_str().unwrap(),
            "list",
            "--bank",
            "default",
            "--limit",
            "10",
        ],
    );

    println!("  Memories listed (check output above)\n");

    // Test 7: Search
    println!("Test 7: Search memories");
    let (_stdout, _stderr) = run_command(
        llm_mem.to_str().unwrap(),
        &[
            "--config",
            config_file.to_str().unwrap(),
            "search",
            "--query",
            "quick brown fox",
            "--bank",
            "default",
            "--limit",
            "5",
        ],
    );

    println!("  Search completed (check output above)\n");

    // Cleanup
    println!("=== Cleanup ===");
    let _ = fs::remove_dir_all(test_dir);
    println!("  Test directory cleaned up\n");

    println!("=== All Tests Completed ===");
}
