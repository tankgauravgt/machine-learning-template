from src.config import MLMConfig
from src.data_pipeline import DataPipeline
from src.model import ModelFactory
from src.trainer import MLMTrainer

def main():
    config = MLMConfig()
    
    print("Initialising Data Pipeline...")
    pipeline = DataPipeline(config)
    dataset_stream = pipeline.load_stream()
    
    print("Building tokeniser...")
    tokenizer = pipeline.build_tokenizer(dataset_stream)
    
    print("Preparing tokenised dataset and collator...")
    # Re-initialise the stream for the mapping phase to ensure the iterator resets
    dataset_stream = pipeline.load_stream() 
    tokenized_dataset, collator = pipeline.prepare_dataset(dataset_stream)
    
    print("Constructing Hugging Face model architecture...")
    model = ModelFactory.create_masked_lm(config, tokenizer)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters : {total_params:,}")
    
    print("Commencing training execution...")
    trainer = MLMTrainer(config, model, tokenized_dataset, collator)
    trainer.execute()
    
    print("Training complete. Model and tokeniser successfully persisted.")

if __name__ == "__main__":
    main()