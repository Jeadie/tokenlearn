//! Per-text encode latency of model2vec-rs models, the call that spiced makes for each text.
//!
//! Usage: ngram_latency <texts.json> <passes> <label=model_dir>...
//!
//! Each pass encodes every text once with each model in turn, one text per call, so the models
//! see the same machine state. Prints one JSON line per model with the sample count and the
//! 50th, 99th and 99.9th percentiles in microseconds.

use std::time::Instant;

use model2vec_rs::model::StaticModel;

fn percentile(sorted: &[u64], q: f64) -> f64 {
    let rank = ((q * sorted.len() as f64).ceil() as usize).clamp(1, sorted.len());
    sorted[rank - 1] as f64 / 1000.0
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    anyhow::ensure!(args.len() >= 4, "usage: ngram_latency <texts.json> <passes> <label=model_dir>...");
    let texts: Vec<String> = serde_json::from_str(&std::fs::read_to_string(&args[1])?)?;
    let passes: usize = args[2].parse()?;
    let mut models = Vec::new();
    for arg in &args[3..] {
        let (label, path) = arg.split_once('=').ok_or_else(|| anyhow::anyhow!("expected label=path: {arg}"))?;
        models.push((label.to_string(), StaticModel::from_pretrained(path, None, None, None)?));
    }

    // One unmeasured pass per model warms caches and the allocator.
    for (_, model) in &models {
        for text in &texts {
            std::hint::black_box(model.encode_single(text));
        }
    }
    let mut samples = vec![Vec::with_capacity(passes * texts.len()); models.len()];
    for _ in 0..passes {
        for (i, (_, model)) in models.iter().enumerate() {
            for text in &texts {
                let start = Instant::now();
                std::hint::black_box(model.encode_single(text));
                samples[i].push(start.elapsed().as_nanos() as u64);
            }
        }
    }
    for ((label, _), mut values) in models.iter().zip(samples) {
        values.sort_unstable();
        println!(
            "{}",
            serde_json::json!({
                "model": label,
                "texts": texts.len(),
                "samples": values.len(),
                "p50_us": percentile(&values, 0.50),
                "p99_us": percentile(&values, 0.99),
                "p99_9_us": percentile(&values, 0.999),
            })
        );
    }
    Ok(())
}
