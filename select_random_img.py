import os 
import random
import shutil

ROOT = "data_preprocessed"
OUT_ROOT = "data_balanced"
TRAIN_SIZE = 1600
TEST_SIZE = 400

def sample_and_copy(split, target_per_class):
    in_split_dir = os.path.join(ROOT, split)
    out_split_dir = os.path.join(OUT_ROOT, split)
    os.makedirs(out_split_dir, exist_ok=True)

    print(f"\n--- Processando split: {split} ---")

    for label in sorted(os.listdir(in_split_dir)):
        in_label_dir = os.path.join(in_split_dir, label)
        if not os.path.isdir(in_label_dir):
            continue

        out_label_dir = os.path.join(out_split_dir, label)
        os.makedirs(out_label_dir, exist_ok=True)

        files = [f for f in os.listdir(in_label_dir) if f.lower().endswith(".png")]

        n_files = len(files)
        print(f"[INFO] {split} - label '{label}': {n_files} imagens disponíveis")

        chosen = random.sample(files, target_per_class)

        for fname in chosen:
            src = os.path.join(in_label_dir, fname)
            dst = os.path.join(out_label_dir, fname)
            shutil.copy2(src, dst)  # preserva metadata básica[web:105][web:108]

        print(f"[DEBUG] {split} - label '{label}': "
              f"{len(chosen)} imagens copiadas para {out_label_dir}")

def main():
    random.seed(42) #reprodutibilidade

    os.makedirs(OUT_ROOT, exist_ok=True)

    sample_and_copy("train", TRAIN_SIZE)
    sample_and_copy("test", TEST_SIZE)

    print("\nConcluído. Estrutura criada em:", OUT_ROOT)

if __name__ == "__main__":
    main()