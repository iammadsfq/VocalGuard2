use crate::{process_batch, LfccConfig};

use parquet::basic::Compression;
use parquet::column::writer::ColumnWriter;
use parquet::data_type::ByteArray;
use parquet::file::properties::WriterProperties;
use parquet::file::writer::SerializedFileWriter;
use parquet::schema::parser::parse_message_type;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use std::collections::HashMap;
use std::error::Error;
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use symphonia::default::{get_codecs, get_probe};

pub const TARGET_SR: usize = 16_000;
pub const TARGET_HW: usize = 224;
pub const SPECTROGRAM_SIZE: usize = 3 * TARGET_HW * TARGET_HW;

#[derive(Clone, Copy, Debug)]
pub enum Corpus {
    La,
    Pa,
    Both,
}

impl Corpus {
    pub fn from_str(value: &str) -> Result<Self, Box<dyn Error>> {
        match value.to_ascii_lowercase().as_str() {
            "la" => Ok(Self::La),
            "pa" => Ok(Self::Pa),
            "both" => Ok(Self::Both),
            _ => Err(format!("invalid corpus: {value}").into()),
        }
    }
}

#[derive(Clone, Debug)]
pub struct SampleRow {
    pub corpus: &'static str,
    pub utt_id: String,
    pub label: String,
    pub split: String,
    pub audio_path: PathBuf,
}

#[derive(Clone, Debug)]
pub struct ConverterConfig {
    pub data_root: PathBuf,
    pub output_root: PathBuf,
    pub corpus: Corpus,
    pub limit: usize,
    pub seed: u64,
    pub la_spoof_target: Option<usize>,
    pub pa_spoof_target: Option<usize>,
    pub la_bonafide_target: Option<usize>,
    pub pa_bonafide_target: Option<usize>,
    pub config: LfccConfig,
}

pub fn run_converter(config: ConverterConfig) -> Result<(), Box<dyn Error>> {
    let mut samples = collect_samples(&config.data_root, config.corpus)?;
    if samples.is_empty() {
        return Err("No usable samples found. Check metadata and audio folders.".into());
    }

    let has_quota_mode = config.la_spoof_target.is_some()
        || config.pa_spoof_target.is_some()
        || config.la_bonafide_target.is_some()
        || config.pa_bonafide_target.is_some();

    if has_quota_mode {
        let (Some(la_spoof_target), Some(pa_spoof_target), Some(la_bonafide_target), Some(pa_bonafide_target)) = (
            config.la_spoof_target,
            config.pa_spoof_target,
            config.la_bonafide_target,
            config.pa_bonafide_target,
        ) else {
            return Err(
                "quota mode requires --la-spoof-target, --pa-spoof-target, --la-bonafide-target, and --pa-bonafide-target"
                    .into(),
            );
        };

        if !matches!(config.corpus, Corpus::Both) {
            return Err("quota mode requires --corpus both".into());
        }

        samples = build_quota_sample(
            samples,
            la_spoof_target,
            pa_spoof_target,
            la_bonafide_target,
            pa_bonafide_target,
            config.seed,
        );
    } else if config.limit > 0 && config.limit < samples.len() {
        let mut rng = StdRng::seed_from_u64(config.seed);
        samples.shuffle(&mut rng);
        samples.truncate(config.limit);
    }

    fs::create_dir_all(&config.output_root)?;

    let manifest_path = config.output_root.join("manifest.csv");
    let failures_path = config.output_root.join("failures.csv");
    let mut manifest = csv::Writer::from_path(&manifest_path)?;
    manifest.write_record([
        "corpus",
        "utt_id",
        "label",
        "label_id",
        "split",
        "audio_path",
        "parquet_path",
    ])?;

    let mut failures = csv::Writer::from_path(&failures_path)?;
    failures.write_record(["utt_id", "error"])?;

    let mut ok_count = 0usize;
    let mut fail_count = 0usize;

    for row in samples {
        let corpus_dir = config.output_root.join(row.corpus);
        fs::create_dir_all(&corpus_dir)?;
        let parquet_path = corpus_dir.join(format!("{}.parquet", row.utt_id));
        match convert_one(&row, &config.config, &parquet_path) {
            Ok(()) => {
                let label_id = if row.label == "bonafide" { "0" } else { "1" };
                manifest.write_record([
                    row.corpus,
                    &row.utt_id,
                    &row.label,
                    label_id,
                    &row.split,
                    &row.audio_path.to_string_lossy(),
                    &parquet_path.to_string_lossy(),
                ])?;
                ok_count += 1;
            }
            Err(err) => {
                failures.write_record([&row.utt_id, &err.to_string()])?;
                fail_count += 1;
            }
        }
    }

    manifest.flush()?;
    failures.flush()?;

    println!("Done. Converted: {ok_count} file(s)");
    println!("Manifest: {}", manifest_path.display());
    if fail_count > 0 {
        println!("Failed: {fail_count} file(s)");
        println!("Failure log: {}", failures_path.display());
    }

    Ok(())
}

fn build_quota_sample(
    all: Vec<SampleRow>,
    la_spoof_target: usize,
    pa_spoof_target: usize,
    la_bonafide_target: usize,
    pa_bonafide_target: usize,
    seed: u64,
) -> Vec<SampleRow> {
    let mut la_spoof = Vec::new();
    let mut pa_spoof = Vec::new();
    let mut la_bonafide = Vec::new();
    let mut pa_bonafide = Vec::new();

    for row in all {
        if row.label == "spoof" && row.corpus == "LA" {
            la_spoof.push(row);
        } else if row.label == "spoof" && row.corpus == "PA" {
            pa_spoof.push(row);
        } else if row.label == "bonafide" && row.corpus == "LA" {
            la_bonafide.push(row);
        } else if row.label == "bonafide" && row.corpus == "PA" {
            pa_bonafide.push(row);
        }
    }

    let la_selected = stratified_sample(
        la_spoof,
        la_spoof_target,
        seed.wrapping_add(101),
        |r| r.split.clone(),
    );
    let pa_selected = stratified_sample(
        pa_spoof,
        pa_spoof_target,
        seed.wrapping_add(202),
        |r| r.split.clone(),
    );
    let la_bonafide_selected = stratified_sample(
        la_bonafide,
        la_bonafide_target,
        seed.wrapping_add(303),
        |r| r.split.clone(),
    );
    let pa_bonafide_selected = stratified_sample(
        pa_bonafide,
        pa_bonafide_target,
        seed.wrapping_add(404),
        |r| r.split.clone(),
    );

    let mut out = Vec::with_capacity(
        la_selected.len()
            + pa_selected.len()
            + la_bonafide_selected.len()
            + pa_bonafide_selected.len(),
    );
    out.extend(la_selected);
    out.extend(pa_selected);
    out.extend(la_bonafide_selected);
    out.extend(pa_bonafide_selected);

    let mut rng = StdRng::seed_from_u64(seed.wrapping_add(505));
    out.shuffle(&mut rng);

    let la_spoof_final = out
        .iter()
        .filter(|r| r.corpus == "LA" && r.label == "spoof")
        .count();
    let pa_spoof_final = out
        .iter()
        .filter(|r| r.corpus == "PA" && r.label == "spoof")
        .count();
    let la_bonafide_final = out
        .iter()
        .filter(|r| r.corpus == "LA" && r.label == "bonafide")
        .count();
    let pa_bonafide_final = out
        .iter()
        .filter(|r| r.corpus == "PA" && r.label == "bonafide")
        .count();

    println!(
        "Quota sampling selected: LA spoof={} | LA bonafide={} | PA spoof={} | PA bonafide={} | total= {}",
        la_spoof_final,
        la_bonafide_final,
        pa_spoof_final,
        pa_bonafide_final,
        out.len()
    );

    out
}

fn stratified_sample<F>(
    rows: Vec<SampleRow>,
    target: usize,
    seed: u64,
    key_fn: F,
) -> Vec<SampleRow>
where
    F: Fn(&SampleRow) -> String,
{
    if target == 0 || rows.is_empty() {
        return Vec::new();
    }

    if target >= rows.len() {
        return rows;
    }

    let mut groups: HashMap<String, Vec<SampleRow>> = HashMap::new();
    for row in rows {
        let key = key_fn(&row);
        groups.entry(key).or_default().push(row);
    }

    let total = groups.values().map(|g| g.len()).sum::<usize>();
    if total == 0 {
        return Vec::new();
    }

    let mut rng = StdRng::seed_from_u64(seed);
    for group in groups.values_mut() {
        group.shuffle(&mut rng);
    }

    let mut allocations: HashMap<String, usize> = HashMap::new();
    let mut fractions = Vec::new();
    let mut used = 0usize;

    for (k, group) in &groups {
        let exact = (target as f64 * group.len() as f64) / total as f64;
        let base = exact.floor() as usize;
        let alloc = base.min(group.len());
        allocations.insert(k.clone(), alloc);
        used += alloc;
        fractions.push((k.clone(), exact - base as f64));
    }

    if used < target {
        fractions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        for (k, _) in &fractions {
            if used >= target {
                break;
            }
            let max_for_group = groups.get(k).map(|g| g.len()).unwrap_or(0);
            let current = *allocations.get(k).unwrap_or(&0);
            if current < max_for_group {
                allocations.insert(k.clone(), current + 1);
                used += 1;
            }
        }
    }

    let mut out = Vec::with_capacity(target);
    let mut leftovers = Vec::new();

    for (k, mut group) in groups {
        let take_n = allocations.get(&k).copied().unwrap_or(0).min(group.len());
        let rest = group.split_off(take_n);
        out.extend(group);
        leftovers.extend(rest);
    }

    if out.len() < target && !leftovers.is_empty() {
        leftovers.shuffle(&mut rng);
        let need = target - out.len();
        out.extend(leftovers.into_iter().take(need));
    }

    if out.len() > target {
        out.shuffle(&mut rng);
        out.truncate(target);
    }

    out
}

pub fn collect_samples(
    data_root: &Path,
    corpus: Corpus,
) -> Result<Vec<SampleRow>, Box<dyn Error>> {
    let mut all = Vec::new();

    match corpus {
        Corpus::La | Corpus::Both => {
            let metadata = data_root.join("LA_keys/LA/CM/trial_metadata.txt");
            let audio_dir = data_root.join("LA_eval/flac");
            all.extend(build_samples_from_metadata(&metadata, &audio_dir, "LA")?);
        }
        Corpus::Pa => {}
    }

    match corpus {
        Corpus::Pa | Corpus::Both => {
            let metadata = data_root.join("PA_keys/PA/CM/trial_metadata.txt");
            let audio_dir = data_root.join("PA_eval/flac");
            all.extend(build_samples_from_metadata(&metadata, &audio_dir, "PA")?);
        }
        Corpus::La => {}
    }

    Ok(all)
}

fn build_samples_from_metadata(
    metadata_path: &Path,
    audio_dir: &Path,
    corpus: &'static str,
) -> Result<Vec<SampleRow>, Box<dyn Error>> {
    let file = File::open(metadata_path)?;
    let reader = BufReader::new(file);
    let mut rows = Vec::new();

    for line in reader.lines() {
        let line = line?;
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 3 {
            continue;
        }

        let utt_id = parts[1].to_string();
        let label = parts
            .iter()
            .find(|p| **p == "bonafide" || **p == "spoof")
            .map(|s| (*s).to_string());

        let split = parts
            .iter()
            .rev()
            .find(|p| **p == "eval" || **p == "progress" || **p == "hidden")
            .map(|s| (*s).to_string())
            .unwrap_or_else(|| "unknown".to_string());

        let Some(label) = label else { continue };
        let audio_path = audio_dir.join(format!("{utt_id}.flac"));
        if !audio_path.exists() {
            continue;
        }

        rows.push(SampleRow {
            corpus,
            utt_id,
            label,
            split,
            audio_path,
        });
    }

    Ok(rows)
}

fn convert_one(
    row: &SampleRow,
    cfg: &LfccConfig,
    parquet_path: &Path,
) -> Result<(), Box<dyn Error>> {
    let samples = decode_flac_mono_16k(&row.audio_path, cfg.sample_rate as u32)?;
    let out = process_batch(&samples, cfg.clone())
        .ok_or("audio too short to produce even one LFCC frame")?;

    if out.data.len() != SPECTROGRAM_SIZE {
        return Err(format!("unexpected tensor size: {}", out.data.len()).into());
    }

    write_parquet_file(parquet_path, &row.utt_id, &out.data)
}

fn write_parquet_file(path: &Path, id: &str, spectrogram: &[f32]) -> Result<(), Box<dyn Error>> {
    let schema = std::sync::Arc::new(parse_message_type(
        "message schema {
            REQUIRED BINARY id (UTF8);
            REQUIRED BINARY spectrogram;
        }",
    )?);

    let props = std::sync::Arc::new(
        WriterProperties::builder()
            .set_compression(Compression::UNCOMPRESSED)
            .build(),
    );

    let file = File::create(path)?;
    let mut writer = SerializedFileWriter::new(file, schema, props)?;
    let mut row_group = writer.next_row_group()?;

    let spectrogram_bytes: Vec<u8> = spectrogram
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();

    let mut column_index = 0usize;
    while let Some(mut column) = row_group.next_column()? {
        match column.untyped() {
            ColumnWriter::ByteArrayColumnWriter(typed) => {
                if column_index == 0 {
                    typed.write_batch(&[ByteArray::from(id)], None, None)?;
                } else {
                    typed.write_batch(&[ByteArray::from(spectrogram_bytes.as_slice())], None, None)?;
                }
            }
            _ => return Err("unexpected parquet column type".into()),
        }
        column.close()?;
        column_index += 1;
    }

    row_group.close()?;
    writer.close()?;
    Ok(())
}

fn decode_flac_mono_16k(path: &Path, target_sr: u32) -> Result<Vec<f32>, Box<dyn Error>> {
    let mut hint = Hint::new();
    hint.with_extension("flac");

    let file = File::open(path)?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    let probed = get_probe().format(
        &hint,
        mss,
        &FormatOptions::default(),
        &MetadataOptions::default(),
    )?;
    let mut format = probed.format;

    let track = format
        .default_track()
        .ok_or("no default track found in audio file")?;

    let sr = track.codec_params.sample_rate.ok_or("missing sample rate")?;
    if sr != target_sr {
        return Err(format!("unsupported sample rate {sr}, expected {target_sr}").into());
    }

    let mut decoder = get_codecs().make(&track.codec_params, &DecoderOptions::default())?;
    let mut mono = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(SymphoniaError::IoError(_)) => break,
            Err(err) => return Err(err.into()),
        };

        let decoded = match decoder.decode(&packet) {
            Ok(decoded) => decoded,
            Err(SymphoniaError::DecodeError(_)) => continue,
            Err(err) => return Err(err.into()),
        };

        let spec = *decoded.spec();
        let channels = spec.channels.count();
        let mut sample_buf = SampleBuffer::<f32>::new(decoded.capacity() as u64, spec);
        sample_buf.copy_interleaved_ref(decoded);
        let interleaved = sample_buf.samples();

        if channels == 1 {
            mono.extend_from_slice(interleaved);
        } else {
            for frame in interleaved.chunks(channels) {
                let sum: f32 = frame.iter().copied().sum();
                mono.push(sum / channels as f32);
            }
        }
    }

    if mono.is_empty() {
        return Err("decoded audio is empty".into());
    }

    Ok(mono)
}
