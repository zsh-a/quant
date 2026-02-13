from src.alpha_mining.persistence import AlphaZooPersistence
import pandas as pd

def check():
    p = AlphaZooPersistence()
    alphas = p.load_all()
    
    if not alphas:
        print("Alpha Zoo is empty. Run the pipeline first.")
        return
        
    print(f"Total Alphas in Zoo: {len(alphas)}")
    print("
Top 5 Factors by RankIC:")
    for i, a in enumerate(alphas[:5]):
        print(f"{i+1}. [{a['id']}] {a['formula']}")
        print(f"   RankIC: {a['metrics'].get('rank_ic'):.4f} | IR: {a['metrics'].get('ic_ir'):.4f}")

if __name__ == "__main__":
    check()
