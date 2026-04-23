//! Performance benchmarks for layered memory operations
//!
//! This benchmark suite measures performance of:
//! - Layer filtering in searches
//! - State filtering
//! - Layer statistics computation
//! - Memory creation with layer metadata

#![cfg(feature = "criterion")]

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use llm_mem::{
    types::{Filters, LayerInfo, Memory, MemoryMetadata, MemoryState, MemoryType},
    vector_store::{VectorLiteConfig, VectorLiteStore, VectorStore},
};
use uuid::Uuid;

/// Create test memories at different layers
fn create_test_memories(store: &VectorLiteStore, count_per_layer: usize) -> Vec<String> {
    let mut ids = Vec::new();

    for layer in 0..=3 {
        for i in 0..count_per_layer {
            let content = format!("Test content for L{} item {}", layer, i);
            let embedding = vec![0.1; 384]; // all-MiniLM-L6-v2 dimension
            let mut metadata = MemoryMetadata::new(MemoryType::Semantic).with_layer(match layer {
                0 => LayerInfo::raw_content(),
                1 => LayerInfo::structural(),
                2 => LayerInfo::semantic(),
                3 => LayerInfo::concept(),
                _ => LayerInfo::default(),
            });

            // Add abstraction sources for higher layers
            if layer > 0 {
                let source_ids: Vec<Uuid> = (0..3).map(|_| Uuid::new_v4()).collect();
                metadata = metadata.with_abstraction_sources(source_ids);
            }

            let memory = Memory::with_content(content, embedding, metadata);
            let id = memory.id.clone();
            futures::executor::block_on(store.insert(&memory)).unwrap();
            ids.push(id);
        }
    }

    ids
}

/// Benchmark layer filtering performance
fn bench_layer_filtering(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let mut group = c.benchmark_group("layer_filtering");
    group.sample_size(50);
    group.measurement_time(std::time::Duration::from_secs(30));

    for size in [100, 500, 1000] {
        let store = rt.block_on(async {
            let config = VectorLiteConfig {
                collection_name: format!("bench_filter_{}", size),
                persistence_path: None,
                ..VectorLiteConfig::default()
            };
            VectorLiteStore::with_config(config).unwrap()
        });

        rt.block_on(async {
            create_test_memories(&store, size / 4);
        });

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _: &usize| {
            b.iter_custom(|iters: u64| {
                let mut total = std::time::Duration::ZERO;

                for _ in 0..iters {
                    let mut filters = Filters::default();
                    filters
                        .custom
                        .insert("layer.level".to_string(), serde_json::json!(1));

                    let start = std::time::Instant::now();
                    rt.block_on(async {
                        let _results = store.list(&filters, None).await.unwrap();
                    });
                    total += start.elapsed();
                }

                total
            })
        });
    }

    group.finish();
}

/// Benchmark state filtering performance
fn bench_state_filtering(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let mut group = c.benchmark_group("state_filtering");
    group.sample_size(50);
    group.measurement_time(std::time::Duration::from_secs(30));

    for size in [100, 500, 1000] {
        let store = rt.block_on(async {
            let config = VectorLiteConfig {
                collection_name: format!("bench_state_{}", size),
                persistence_path: None,
                ..VectorLiteConfig::default()
            };
            VectorLiteStore::with_config(config).unwrap()
        });

        rt.block_on(async {
            let ids = create_test_memories(&store, size / 4);
            // Mark some as forgotten
            for (i, id) in ids.iter().enumerate() {
                if i % 5 == 0 {
                    let mut memory = store.get(id).await.unwrap().unwrap();
                    memory.metadata.state = MemoryState::Forgotten;
                    store.update(&memory).await.unwrap();
                }
            }
        });

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _: &usize| {
            b.iter_custom(|iters: u64| {
                let mut total = std::time::Duration::ZERO;

                for _ in 0..iters {
                    let mut filters = Filters::default();
                    filters
                        .custom
                        .insert("state".to_string(), serde_json::json!("Active"));

                    let start = std::time::Instant::now();
                    rt.block_on(async {
                        let _results = store.list(&filters, None).await.unwrap();
                    });
                    total += start.elapsed();
                }

                total
            })
        });
    }

    group.finish();
}

/// Benchmark layer statistics computation
fn bench_layer_stats(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let mut group = c.benchmark_group("layer_stats");
    group.sample_size(50);
    group.measurement_time(std::time::Duration::from_secs(30));

    for size in [100, 500, 1000, 5000] {
        let store = rt.block_on(async {
            let config = VectorLiteConfig {
                collection_name: format!("bench_stats_{}", size),
                persistence_path: None,
                ..VectorLiteConfig::default()
            };
            VectorLiteStore::with_config(config).unwrap()
        });

        rt.block_on(async {
            create_test_memories(&store, size / 4);
        });

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _: &usize| {
            b.iter_custom(|iters: u64| {
                let mut total = std::time::Duration::ZERO;

                for _ in 0..iters {
                    let start = std::time::Instant::now();
                    rt.block_on(async {
                        let memories = store.list(&Filters::default(), None).await.unwrap();

                        // Compute stats
                        let mut layer_counts = std::collections::HashMap::new();
                        let mut state_counts = std::collections::HashMap::new();
                        for memory in &memories {
                            *layer_counts
                                .entry(memory.metadata.layer.level)
                                .or_insert(0usize) += 1;
                            *state_counts
                                .entry(format!("{:?}", memory.metadata.state))
                                .or_insert(0usize) += 1;
                        }
                        black_box((layer_counts, state_counts));
                    });
                    total += start.elapsed();
                }

                total
            })
        });
    }

    group.finish();
}

/// Benchmark memory creation with layer metadata
fn bench_memory_creation(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let mut group = c.benchmark_group("memory_creation");
    group.sample_size(100);
    group.measurement_time(std::time::Duration::from_secs(20));

    for layer in [0, 1, 2, 3] {
        let store = rt.block_on(async {
            let config = VectorLiteConfig {
                collection_name: format!("bench_create_l{}", layer),
                persistence_path: None,
                ..VectorLiteConfig::default()
            };
            VectorLiteStore::with_config(config).unwrap()
        });

        group.bench_with_input(
            BenchmarkId::new("layer", layer),
            &layer,
            |b, &layer: &usize| {
                b.iter_custom(|iters: u64| {
                    let mut total = std::time::Duration::ZERO;

                    for _ in 0..iters {
                        let content = format!("Test content for L{}", layer);
                        let embedding = vec![0.1; 384];
                        let metadata =
                            MemoryMetadata::new(MemoryType::Semantic).with_layer(match layer {
                                0 => LayerInfo::raw_content(),
                                1 => LayerInfo::structural(),
                                2 => LayerInfo::semantic(),
                                3 => LayerInfo::concept(),
                                _ => LayerInfo::default(),
                            });

                        let memory = Memory::with_content(content, embedding, metadata);

                        let start = std::time::Instant::now();
                        rt.block_on(async {
                            store.insert(&memory).await.unwrap();
                        });
                        total += start.elapsed();
                    }

                    total
                })
            },
        );
    }

    group.finish();
}

/// Benchmark combined layer + type filtering
fn bench_combined_filtering(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let mut group = c.benchmark_group("combined_filtering");
    group.sample_size(50);
    group.measurement_time(std::time::Duration::from_secs(30));

    for size in [100, 500, 1000] {
        let store = rt.block_on(async {
            let config = VectorLiteConfig {
                collection_name: format!("bench_combined_{}", size),
                persistence_path: None,
                ..VectorLiteConfig::default()
            };
            VectorLiteStore::with_config(config).unwrap()
        });

        rt.block_on(async {
            let memories = create_test_memories(&store, size / 4);
            // Vary memory types
            for (i, id) in memories.iter().enumerate() {
                let mut memory = store.get(id).await.unwrap().unwrap();
                memory.metadata.memory_type = match i % 6 {
                    0 => MemoryType::Conversational,
                    1 => MemoryType::Procedural,
                    2 => MemoryType::Factual,
                    3 => MemoryType::Semantic,
                    4 => MemoryType::Episodic,
                    _ => MemoryType::Personal,
                };
                store.update(&memory).await.unwrap();
            }
        });

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _: &usize| {
            b.iter_custom(|iters: u64| {
                let mut total = std::time::Duration::ZERO;

                for _ in 0..iters {
                    let filters = Filters {
                        memory_type: Some(MemoryType::Semantic),
                        custom: {
                            let mut map = std::collections::HashMap::new();
                            map.insert("layer.level".to_string(), serde_json::json!(2));
                            map
                        },
                        ..Default::default()
                    };

                    let start = std::time::Instant::now();
                    rt.block_on(async {
                        let _results = store.list(&filters, None).await.unwrap();
                    });
                    total += start.elapsed();
                }

                total
            })
        });
    }

    group.finish();
}

#[cfg(feature = "criterion")]
criterion_group!(
    benches,
    bench_layer_filtering,
    bench_state_filtering,
    bench_layer_stats,
    bench_memory_creation,
    bench_combined_filtering,
);

#[cfg(feature = "criterion")]
criterion_main!(benches);

#[cfg(not(feature = "criterion"))]
fn main() {}
