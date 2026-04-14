use crate::OutputFormat;

/// Format bytes into a human-readable string (e.g. "4.2 GB", "512 MB")
fn format_bytes(bytes: usize) -> String {
    const KB: f64 = 1024.0;
    const MB: f64 = KB * 1024.0;
    const GB: f64 = MB * 1024.0;
    let b = bytes as f64;
    if b >= GB {
        format!("{:.1} GB", b / GB)
    } else if b >= MB {
        format!("{:.0} MB", b / MB)
    } else if b >= KB {
        format!("{:.0} KB", b / KB)
    } else {
        format!("{} B", bytes)
    }
}

/// Handle the list-devices command
pub fn handle_list_devices(format: OutputFormat) -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(feature = "local")]
    {
        use llama_cpp_2::LlamaBackendDeviceType;

        let devices = llama_cpp_2::list_llama_ggml_backend_devices();

        let gpu_devices: Vec<_> = devices
            .iter()
            .filter(|d| matches!(
                d.device_type,
                LlamaBackendDeviceType::Gpu
                    | LlamaBackendDeviceType::IntegratedGpu
                    | LlamaBackendDeviceType::Accelerator
            ))
            .collect();

        match format {
            OutputFormat::Json | OutputFormat::Jsonl => {
                let devices_json: Vec<serde_json::Value> = devices
                    .iter()
                    .map(|d| {
                        let type_str = match d.device_type {
                            LlamaBackendDeviceType::Gpu => "GPU",
                            LlamaBackendDeviceType::IntegratedGpu => "iGPU",
                            LlamaBackendDeviceType::Accelerator => "Accelerator",
                            LlamaBackendDeviceType::Cpu => "CPU",
                            _ => "Unknown",
                        };
                        serde_json::json!({
                            "index": d.index,
                            "name": d.name,
                            "description": d.description,
                            "type": type_str,
                            "memory_total": d.memory_total,
                            "memory_free": d.memory_free,
                            "memory_total_human": format_bytes(d.memory_total),
                            "memory_free_human": format_bytes(d.memory_free),
                        })
                    })
                    .collect();
                println!("{}", serde_json::to_string_pretty(&devices_json)?);
            }
            _ => {
                // Table / Detail / CSV — all use the same text layout
                println!("llama.cpp backend devices ({} total):\n", devices.len());
                for dev in &devices {
                    let type_label = match dev.device_type {
                        LlamaBackendDeviceType::Gpu => "GPU",
                        LlamaBackendDeviceType::IntegratedGpu => "iGPU",
                        LlamaBackendDeviceType::Accelerator => "Accelerator",
                        LlamaBackendDeviceType::Cpu => "CPU",
                        _ => "Unknown",
                    };
                    println!(
                        "  [{}] {} — {} ({} free / {} total) [{}]",
                        dev.index,
                        dev.name,
                        dev.description,
                        format_bytes(dev.memory_free),
                        format_bytes(dev.memory_total),
                        type_label,
                    );
                }

                if gpu_devices.is_empty() {
                    println!("\nNo GPU devices found. GPU offload is not available.");
                    println!("  Build with Vulkan: cargo build --no-default-features --features \"local,vulkan\"");
                } else {
                    println!(
                        "\nGPU devices available: {} — set gpu_layers > 0 in config or LLM_MEM_GPU_LAYERS to use",
                        gpu_devices.iter().map(|d| d.name.as_str()).collect::<Vec<_>>().join(", ")
                    );
                }
            }
        }
    }

    #[cfg(not(feature = "local"))]
    {
        match format {
            OutputFormat::Json | OutputFormat::Jsonl => {
                println!("{}", serde_json::to_string_pretty(&serde_json::json!({
                    "devices": [],
                    "error": "Local inference not enabled. Build with --features local,vulkan"
                }))?);
            }
            _ => {
                println!("Local inference (llama.cpp) is not enabled in this build.");
                println!("Rebuild with: cargo build --features local,vulkan");
            }
        }
    }

    Ok(())
}
