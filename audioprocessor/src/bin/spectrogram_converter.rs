use audioprocessor::converter::{run_converter, ConverterConfig, Corpus};
use audioprocessor::LfccConfig;
use clap::Parser;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "spectrogram_converter", version, about = "Convert ASVspoof audio to parquet spectrograms")]
struct Cli {
    #[arg(long, default_value = "data")]
    data_root: PathBuf,
    #[arg(long, default_value = "data/spectrogram")]
    output_root: PathBuf,
    #[arg(long, default_value = "both")]
    corpus: String,
    #[arg(long, default_value_t = 50)]
    limit: usize,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    #[arg(long)]
    la_spoof_target: Option<usize>,
    #[arg(long)]
    pa_spoof_target: Option<usize>,
    #[arg(long)]
    la_bonafide_target: Option<usize>,
    #[arg(long)]
    pa_bonafide_target: Option<usize>,
}

fn main() {
    let cli = Cli::parse();
    let corpus = match Corpus::from_str(&cli.corpus) {
        Ok(corpus) => corpus,
        Err(err) => {
            eprintln!("Error: {err}");
            std::process::exit(1);
        }
    };

    let config = ConverterConfig {
        data_root: cli.data_root,
        output_root: cli.output_root,
        corpus,
        limit: cli.limit,
        seed: cli.seed,
        la_spoof_target: cli.la_spoof_target,
        pa_spoof_target: cli.pa_spoof_target,
        la_bonafide_target: cli.la_bonafide_target,
        pa_bonafide_target: cli.pa_bonafide_target,
        config: LfccConfig::default(),
    };

    if let Err(err) = run_converter(config) {
        eprintln!("Error: {err}");
        std::process::exit(1);
    }
}
