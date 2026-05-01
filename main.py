import argparse
from src.d_tuning import run_multi_tuning
from src.e_train import run_final_training
from src.f_evaluate import run_evaluation

def main():
    # Membuat interface command-line
    parser = argparse.ArgumentParser(description="Pipeline Machine Learning Churn Prediction")
    
    # Menambahkan pilihan mode agar fleksibel
    parser.add_argument(
        '--mode', 
        choices=['tuning', 'train', 'evaluate', 'all'], 
        default='all',
        help="Pilih mode eksekusi: 'tuning', 'train', 'evaluate', atau 'all' (default)"
    )
    
    args = parser.parse_args()

    print(f"--- Memulai mode: {args.mode.upper()} ---")

    # Logika eksekusi berdasarkan input user
    if args.mode in ['tuning', 'all']:
        print(">> Tahap: Hyperparameter Tuning")
        run_multi_tuning()
        
    if args.mode in ['train', 'all']:
        print(">> Tahap: Training Final")
        run_final_training()
        
    if args.mode in ['evaluate', 'all']:
        print(">> Tahap: Evaluasi Model")
        run_evaluation()

    print("--- Selesai ---")

if __name__ == "__main__":
    main()