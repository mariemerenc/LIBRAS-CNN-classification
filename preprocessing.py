import os
import cv2

INPUT_ROOT = "data"
OUTPUT_ROOT = "data_preprocessed"
TARGET_SIZE = (64, 64)

def preprocess_opencv(path):
    img = cv2.imread(path, 0)
    if img is None:
        print(f"[WARN] falha ao ler imagem: {path}")
        return None
    
    gray = cv2.GaussianBlur(img, (5, 5), 0)
    
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        hand = img[y:y+h, x:x+w]
    else:
        hand = img
    
    hand = cv2.resize(hand, TARGET_SIZE)
    return hand


def process_split(split_name):
    input_split_dir = os.path.join(INPUT_ROOT, split_name)
    output_split_dir = os.path.join(OUTPUT_ROOT, split_name)
    
    #dict p/ contar imagens por label
    counts = {}
    
    for label in sorted(os.listdir(input_split_dir)):
        input_label_dir = os.path.join(input_split_dir, label)
        if not os.path.isdir(input_label_dir):
            continue
        
        output_label_dir = os.path.join(output_split_dir, label)
        os.makedirs(output_label_dir, exist_ok=True)
        
        counts[label] = 0
        
        for fname in os.listdir(input_label_dir):
            in_path = os.path.join(input_label_dir, fname)
            
            if not fname.lower().endswith(".png"):
                continue
            
            img_proc = preprocess_opencv(in_path)
            if img_proc is None:
                continue
            
            out_path = os.path.join(output_label_dir, fname)
            cv2.imwrite(out_path, img_proc)
            counts[label]+=1
        
        print(f"[DEBUG] {split_name} - label '{label}': {counts[label]} imagens processadas")
        
    
    print(f"\n[RESUMO!!] Split: {split_name}")
    total = 0
    
    for label, c in counts.items():
        print(f"{label}: {c} imagens")
        total += c
    print(f"TOTAL {split_name}: {total} imagens\n")
    
    
def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    print("--- Processando 'train' ---")
    process_split("train")

    print("--- Processando 'test' ---")
    process_split("test")


if __name__ == "__main__":
    main()