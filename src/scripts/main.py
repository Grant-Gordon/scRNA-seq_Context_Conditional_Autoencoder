from trainer import Trainer
import scripts.metadata_preprocessor 
import time
import argparse


def main(raw_args=None):
    t0_start_main = time.time()
    #############################
    #Get Job_dir/output_dir from slurm submission script
    parser = argparse.ArgumentParser(description="Parser to take in Output_dir from surm_submission scripts")
    parser.add_argument("--output-dir", required=True, help="Output directory for: logs, *.err, *.out, tensorboard, etc.")
    args = parser.parse_args(raw_args)


    #############################
    #Data
    OUTPUT_DIR=args.output_dir
    DATA_DIR="/mnt/projects/debruinz_project/july2024_census_data/subset"
    META_GLOB="human_metadata_*.pkl"
    EXPR_GLOB="human_counts_*.npz" 
    
    #Preprocessed metadata 
    RERUN_PREPROCESSOR=True
    PREPROCESSOR_DIR="/mnt/home/gordongr/research/scRNA-seq_Context_Conditional_Autoencoder/Preprocessed_metadata"
    META_FIELDS_VOCABS_FILE_NAME="metadata_vocab_subset_default_9.json"
    FIELD_SPECS_FILE_NAME="metadata_field_specs_subset_default_9.json"
    META_FIELDS_VOCABS_PATH=f"{PREPROCESSOR_DIR}/{META_FIELDS_VOCABS_FILE_NAME}"  # { field_name: { value: idx, ... }, ... }   
    FIELD_SPECS_PATH=f"{PREPROCESSOR_DIR}/{FIELD_SPECS_FILE_NAME}"      #[ FieldSpec(field=..., cardinality=..., using=..., non_null_fraction=...), ... ]
    INCLUDE_FIELDS=[
        "cell_type",
        "disease",
        "development_stage",
        "dev_stage",
        "sex",
        "self_reported_ethnicity",
        "tissue_general",
        "tissue",
        "assay"
    ]
    PREPROCESSOR_ARGS=[
        '--data-dir', f'{DATA_DIR}',
        '--pattern', f'{META_GLOB}',
        '--save-dir', f'{PREPROCESSOR_DIR}',
        '--vocab-json-name', f'{META_FIELDS_VOCABS_FILE_NAME}',
        '--specs-json-name', f'{FIELD_SPECS_FILE_NAME}',
        '--verbose'
        ]
    for f in INCLUDE_FIELDS:
        PREPROCESSOR_ARGS+= ["--include", f]

    #Training
    LEARNING_RATE=0.0001
    BATCH_SIZE=128
    NUM_EPOCHS=10
    BATCH_WORKERS=2
    BATCH_PREFETCH_FACTOR=3
    #Model
    LATENT_DIM=128

    #############################
    if RERUN_PREPROCESSOR:
        t0_preprocess_metadata=time.time()
        print(f"RERUN_PROCESSOR:{RERUN_PREPROCESSOR}")
        scripts.metadata_preprocessor.main(PREPROCESSOR_ARGS)
        t1_preprocess_metadata = time.time() - t0_preprocess_metadata
        print(f"[Time Preprocessing]: {t1_preprocess_metadata}, [Time Current]: {time.time()}")

    else:
        print(f"Metdata preprocessor was not used. \n\t Using preprocessed JSONS at:\n\t - {FIELD_SPECS_PATH}\n\t - {META_FIELDS_VOCABS_PATH}")

    trainer = Trainer(
        output_dir=OUTPUT_DIR,
        data_dir=DATA_DIR,
        expr_glob=EXPR_GLOB,
        meta_glob=META_GLOB,
        field_specs_path=FIELD_SPECS_PATH,
        meta_fields_vocabs_path=META_FIELDS_VOCABS_PATH,
        learning_rate=LEARNING_RATE,
        batch_size = BATCH_SIZE,
        latent_dim = LATENT_DIM,
        classifier_latent_dim=LATENT_DIM,
        batch_workers=BATCH_WORKERS,
        batch_prefetch_factor=BATCH_PREFETCH_FACTOR
    )
    t0_start_training = time.time()
    print(f"[Time Until Training]: {time.time() - t0_start_main}, [Time Current]: {time.time()}")
    trainer.train(num_epochs=NUM_EPOCHS)
   
#    time.strftime('%H:%M:%S', time.gmtime(12345)) #TODO: easier formating??
    
    print(f"[Time Spent Training - HHH.MM.SS]: {(train_time:= int(time.time() - t0_start_training))//3600:02d}.{(train_time%3600)//60:02d}.{train_time%60:02d} [Time Current]: {time.time():.2f}")
    print(f"[Time Total - HHH.MM.SS]: {(run_time:= int(time.time() - t0_start_main))//3600:02d}.{(run_time%3600)//60:02d}.{run_time%60:02d} [Time Current]: {time.time():.2f}")



if __name__=="__main__":
    main()