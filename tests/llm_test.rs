#![cfg(feature = "local")]

use llm_mem::config::Config;
use llm_mem::llm::create_llm_client;
use std::time::Instant;

/// Test 1: Can the LLM model load at all?
#[tokio::test]
async fn llm_load_and_health() {
    let mut config = Config::default();
    config.apply_env_overrides();
    if config.llm.cpu_threads <= 0 {
        config.llm.cpu_threads = std::thread::available_parallelism()
            .map(|n| n.get() as i32)
            .unwrap_or(8);
    }
    if config.llm.gpu_layers == 0 {
        config.llm.gpu_layers = 999;
    }

    println!("\n  ── LLM Load Test ──");
    println!(
        "  Model: {} | CPU threads: {} | GPU layers: {}",
        config.llm.model_file, config.llm.cpu_threads, config.llm.gpu_layers
    );

    let client = match create_llm_client(&config).await {
        Ok(c) => c,
        Err(e) => {
            panic!("Client creation failed: {}", e);
        }
    };

    let status = client.get_status();
    println!(
        "  Backend: {} | State: {} | Model: {} | Embedding: {}",
        status.backend, status.state, status.llm_model, status.embedding_model
    );

    // Trigger actual model load by making a real completion call.
    // LazyLocalLLMClient defers loading until the first LLM request.
    let t0 = Instant::now();
    match client.complete("Hello").await {
        Ok(text) => {
            println!("  ✓ Model loaded and generated in {:.1}s", t0.elapsed().as_secs_f32());
            println!("  Response: {:?}", &text[..text.len().min(120)]);
        }
        Err(e) => {
            panic!("Model loaded (health check passed) but completion FAILED: {}", e);
        }
    }

    println!("  ✓ LLM load test passed\n");
}

/// Test 2: Can the LLM produce valid JSON? (most critical for our pipeline)
#[tokio::test]
async fn llm_json_generation() {
    let mut config = Config::default();
    config.apply_env_overrides();
    if config.llm.cpu_threads <= 0 {
        config.llm.cpu_threads = std::thread::available_parallelism()
            .map(|n| n.get() as i32)
            .unwrap_or(8);
    }
    if config.llm.gpu_layers == 0 {
        config.llm.gpu_layers = 999;
    }

    println!("\n  ── JSON Generation Test ──");

    let client = create_llm_client(&config).await
        .expect("Client creation failed");

    // Exact same prompt format as build_l1_prompt
    let prompt = r#"You are creating a structural abstraction of the following content.

SOURCE MEMORY (L0):
In 1928, Alexander Fleming discovered penicillin when mold contaminated his petri dishes.

TASK: Generate a concise summary that:
1. Captures the main topic in 1-2 sentences
2. Identifies the document structure (if applicable): chapter, section, subsection
3. Notes any key entities mentioned

OUTPUT FORMAT: Return exactly a valid JSON object matching this schema:
{
  "summary": "2-3 sentence summary",
  "structure_type": "chunk|section|chapter|document",
  "key_entities": ["entity1", "entity2"],
  "suggested_title": "Brief descriptive title",
  "confidence": 0.95
}
"#;

    let t0 = Instant::now();
    match client.complete(prompt).await {
        Ok(text) => {
            println!("  Generated in {:.1}s ({} bytes)", t0.elapsed().as_secs_f32(), text.len());
            println!("  Raw response:");
            println!("  ─────────────");
            for line in text.lines() {
                println!("  | {}", line);
            }
            println!("  ─────────────");

            // Try to extract JSON and parse it
            let json_parse = llm_mem::llm::extract_json_from_text(&text)
            .and_then(|json_str| serde_json::from_str::<serde_json::Value>(&json_str).ok());

            match json_parse {
                Some(json) => {
                    println!("  ✓ JSON parse SUCCEEDED");
                    if let Some(s) = json.get("summary") {
                        println!("    summary: {:?}", s);
                    }
                    if let Some(s) = json.get("structure_type") {
                        println!("    structure_type: {:?}", s);
                    }
                    if let Some(e) = json.get("key_entities") {
                        println!("    key_entities: {:?}", e);
                    }
                }
                None => {
                    println!("  ✗ JSON parse FAILED");
                    println!("  Attempting to find JSON boundaries manually...");
                    let find_json = text.find('{').and_then(|start| {
                        let mut depth = 1u32;
                        for (i, ch) in text[start + 1..].char_indices() {
                            match ch {
                                '{' => depth += 1,
                                '}' => {
                                    depth -= 1;
                                    if depth == 0 { return Some(&text[start..=start + 1 + i]); }
                                }
                                _ => {}
                            }
                        }
                        None
                    });
                    match find_json {
                        Some(j) => println!("  Found JSON-like content:\n  {}", j),
                        None => println!("  No JSON-like content found at all"),
                    }
                }
            }
        }
        Err(e) => {
            panic!("Completion FAILED: {}", e);
        }
    }

    println!("  ✓ JSON generation test completed\n");
}

/// Test 3: Can the LLM handle multi-source synthesis (L2 prompt format)?
#[tokio::test]
async fn llm_multi_source_synthesis() {
    let mut config = Config::default();
    config.apply_env_overrides();
    if config.llm.cpu_threads <= 0 {
        config.llm.cpu_threads = std::thread::available_parallelism()
            .map(|n| n.get() as i32)
            .unwrap_or(8);
    }
    if config.llm.gpu_layers == 0 {
        config.llm.gpu_layers = 999;
    }

    println!("\n  ── Multi-Source Synthesis Test (L2 format) ──");

    let client = create_llm_client(&config).await
        .expect("Client creation failed");

    // Simulated L1 summaries
    let prompt = r#"You are synthesizing several L1 summaries to create an L2 semantic abstraction. Look for connections and themes across these memories.

SOURCE L1 MEMORIES:
MEMORY 1:
Alexander Fleming discovered penicillin in 1928 when a mold spore contaminated a Staphylococcus culture plate. The mold, later identified as Penicillium notatum, created a bacteria-free zone. This accidental observation led to the world's first antibiotic.

MEMORY 2:
In 1968, Spencer Silver at 3M accidentally created a weak adhesive while trying to develop a super-strong one. Years later, his colleague Art Fry used it to create sticky notes that wouldn't fall out of his hymnal, leading to Post-it Notes.

MEMORY 3:
Percy Spencer discovered microwave cooking in 1945 when a radar magnetron melted a chocolate bar in his pocket. This accidental observation led to the development of the microwave oven.

TASK: Generate a meaningful semantic synthesis that:
1. Identifies the overarching theme or conclusion across these memories.
2. Extracts facts or assertions that span multiple memories.
3. Groups related entities together.

OUTPUT FORMAT: Return exactly a valid JSON object matching this schema:
{
  "synthesis": "A coherent synthesis paragraph",
  "theme": "The main theme connecting them",
  "shared_entities": ["entity1", "entity2"],
  "confidence": 0.85
}
"#;

    let t0 = Instant::now();
    match client.complete(prompt).await {
        Ok(text) => {
            println!("  Generated in {:.1}s ({} bytes)", t0.elapsed().as_secs_f32(), text.len());
            println!("  Raw response:");
            println!("  ─────────────");
            for line in text.lines() {
                println!("  | {}", line);
            }
            println!("  ─────────────");

            let json_parse = llm_mem::llm::extract_json_from_text(&text)
            .and_then(|json_str| serde_json::from_str::<serde_json::Value>(&json_str).ok());

            match json_parse {
                Some(json) => {
                    println!("  ✓ JSON parse SUCCEEDED");
                    if let Some(s) = json.get("synthesis") {
                        println!("    synthesis preview: {:?}", s.as_str().unwrap_or("?").chars().take(80).collect::<String>());
                    }
                    if let Some(s) = json.get("theme") {
                        println!("    theme: {:?}", s);
                    }
                }
                None => {
                    println!("  ✗ JSON parse FAILED");
                }
            }
        }
        Err(e) => {
            panic!("Completion FAILED: {}", e);
        }
    }

    println!("  ✓ Multi-source synthesis test completed\n");
}

/// Test 4: Quick temperature/format variation test
#[tokio::test]
async fn llm_temperature_variation() {
    let mut config = Config::default();
    config.apply_env_overrides();
    if config.llm.cpu_threads <= 0 {
        config.llm.cpu_threads = std::thread::available_parallelism()
            .map(|n| n.get() as i32)
            .unwrap_or(8);
    }
    if config.llm.gpu_layers == 0 {
        config.llm.gpu_layers = 999;
    }

    println!("\n  ── Temperature Variation Test ──");

    let client = create_llm_client(&config).await
        .expect("Client creation failed");

    let prompt = "List three fruits, separated by commas. No explanation. Just: apple, banana, cherry";

    let t0 = Instant::now();
    match client.complete(prompt).await {
        Ok(text) => {
            println!("  Generated in {:.1}s", t0.elapsed().as_secs_f32());
            println!("  Response: {:?}", text.trim());
        }
        Err(e) => {
            println!("  ✗ Completion FAILED: {}", e);
        }
    }

    println!("  ✓ Temperature test completed\n");
}

/// Test 5: Check what chat template / special tokens the model is using
#[tokio::test]
async fn llm_token_behavior() {
    let mut config = Config::default();
    config.apply_env_overrides();
    if config.llm.cpu_threads <= 0 {
        config.llm.cpu_threads = std::thread::available_parallelism()
            .map(|n| n.get() as i32)
            .unwrap_or(8);
    }
    if config.llm.gpu_layers == 0 {
        config.llm.gpu_layers = 999;
    }

    println!("\n  ── Token Behavior Test ──");

    let client = create_llm_client(&config).await
        .expect("Client creation failed");

    // Test with a simple echo instruction to see what the model outputs
    // (some models output their own chat template tokens)
    let prompt = "Repeat exactly this phrase and nothing else: OK";

    let t0 = Instant::now();
    match client.complete(prompt).await {
        Ok(text) => {
            println!("  Generated in {:.1}s", t0.elapsed().as_secs_f32());
            let stripped = text.trim();
            println!(
                "  Response ({} chars): {:?}",
                stripped.len(),
                &stripped[..stripped.len().min(200)]
            );
            // Check if the model is echoing ChatML tokens
            let has_im_start = text.contains("<|im_start|>");
            let has_im_end = text.contains("<|im_end|>");
            let has_bot = text.contains("<start_of_turn>");
            let has_eot = text.contains("<end_of_turn>");
            let has_think = text.contains("<think>");
            println!(
                "  Found tokens: im_start={} im_end={} start_of_turn={} end_of_turn={} think={}",
                has_im_start, has_im_end, has_bot, has_eot, has_think
            );
        }
        Err(e) => {
            println!("  ✗ Completion FAILED: {}", e);
        }
    }

    println!("  ✓ Token behavior test completed\n");
}
